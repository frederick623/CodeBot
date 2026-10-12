use anyhow::Result;
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::{Arc, Once};
use parking_lot::Mutex;

/// Above this many chunks brute-force cosine starts to hurt; the sqlite-vec
/// `vec0` index handles KNN natively from here on (LanceDB/FAISS are the other
/// usual escalations). Kept as documentation for the chosen threshold.
pub const ANN_SCALE_HINT: usize = 50_000;

static VEC_INIT: Once = Once::new();

/// Register the sqlite-vec loadable extension as an auto-extension so every
/// subsequently opened connection exposes the `vec0` virtual table and the
/// `vec_*` SQL functions. Idempotent.
fn register_sqlite_vec() {
    VEC_INIT.call_once(|| unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    });
}

#[derive(Clone)]
pub struct Store {
    conn: Arc<Mutex<Connection>>,
    /// Whether the sqlite-vec extension loaded successfully for this DB.
    vec_enabled: bool,
    /// Dimensionality of the `vec0` table once created (it is fixed at creation).
    vec_dim: Arc<Mutex<Option<usize>>>,
}

#[derive(Debug, Clone)]
pub struct ChunkRow {
    pub id: i64,
    pub file: String,
    pub start_line: usize,
    pub end_line: usize,
    pub symbol: Option<String>,
    pub content: String,
}

/// A frozen entry in the interface registry produced by the planning gate. The
/// canonical `name` is unique per plan and is the source of truth that later
/// implementation steps must conform to.
#[derive(Debug, Clone)]
pub struct PlanSymbol {
    pub module: String,
    pub name: String,
    pub kind: String,
    pub signature: String,
}

/// A unit of implementation work with an explicit lifecycle. The store, not the
/// model, owns the truth about what exists and whether it is verified.
/// `status` is one of: pending | generating | verifying | verified | failed.
#[derive(Debug, Clone)]
pub struct TaskRow {
    pub id: i64,
    pub plan_id: String,
    pub name: String,
    pub status: String,
    pub code: Option<String>,
    pub attempts: i64,
}

impl Store {
    pub fn open(db_path: &Path) -> Result<Self> {
        register_sqlite_vec(); // must run before the connection is opened
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            r#"
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS files (
                path        TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                size        INTEGER NOT NULL,
                mtime_ms    INTEGER NOT NULL,
                indexed_ms  INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY,
                file        TEXT NOT NULL,
                start_line  INTEGER NOT NULL,
                end_line    INTEGER NOT NULL,
                symbol      TEXT,
                content     TEXT NOT NULL,
                FOREIGN KEY(file) REFERENCES files(path) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id    INTEGER PRIMARY KEY,
                dim         INTEGER NOT NULL,
                vec         BLOB NOT NULL,
                FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS symbols (
                id          INTEGER PRIMARY KEY,
                file        TEXT NOT NULL,
                name        TEXT NOT NULL,
                kind        TEXT NOT NULL,
                start_line  INTEGER NOT NULL,
                end_line    INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS edges (
                src_symbol  TEXT NOT NULL,
                dst_symbol  TEXT NOT NULL,
                kind        TEXT NOT NULL   -- "import" | "call" | "reference"
            );
            CREATE TABLE IF NOT EXISTS plan_symbols (
                id        INTEGER PRIMARY KEY,
                plan_id   TEXT NOT NULL,
                module    TEXT NOT NULL,
                name      TEXT NOT NULL,
                kind      TEXT NOT NULL,
                signature TEXT NOT NULL,
                UNIQUE(plan_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file);
            CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
            CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_symbol);
            CREATE TABLE IF NOT EXISTS tasks (
                id        INTEGER PRIMARY KEY,
                plan_id   TEXT NOT NULL,
                name      TEXT NOT NULL,
                status    TEXT NOT NULL,
                code      TEXT,
                attempts  INTEGER NOT NULL DEFAULT 0,
                UNIQUE(plan_id, name)
            );
            CREATE TABLE IF NOT EXISTS errors (
                id        INTEGER PRIMARY KEY,
                task_id   INTEGER NOT NULL,
                attempt   INTEGER NOT NULL,
                checker   TEXT NOT NULL,
                message   TEXT NOT NULL,
                resolved  INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_plan_symbols_plan ON plan_symbols(plan_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_plan ON tasks(plan_id);
            CREATE INDEX IF NOT EXISTS idx_errors_task ON errors(task_id);
            -- Persistent key/value configuration (LLM endpoint, model, …).
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            -- Conversation history, one row per turn, scoped to a workspace root.
            CREATE TABLE IF NOT EXISTS messages (
                id        INTEGER PRIMARY KEY,
                root      TEXT NOT NULL,
                role      TEXT NOT NULL,   -- "user" | "assistant"
                content   TEXT NOT NULL,
                created_ms INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_root ON messages(root, id);
            -- Durable facts the agent has learned about a workspace (what was
            -- edited and why, project conventions, …). Injected into every
            -- prompt so a weak model does not have to rediscover them.
            CREATE TABLE IF NOT EXISTS memory (
                id        INTEGER PRIMARY KEY,
                root      TEXT NOT NULL,
                note      TEXT NOT NULL,
                created_ms INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_root ON memory(root, id);
            "#,
        )?;
        // The durable embedding store is the `embeddings` BLOB table above. When
        // sqlite-vec is present we additionally maintain a `vec0` index for
        // native KNN; the BLOB table remains the source of truth and the
        // brute-force fallback path.
        let vec_enabled = conn
            .query_row("SELECT vec_version()", [], |r| r.get::<_, String>(0))
            .is_ok();
        // Recover the index dimensionality if a vec0 table already exists from a
        // previous run (its byte length is dim * 4 for float32 vectors).
        let vec_dim = if vec_enabled {
            conn.query_row(
                "SELECT length(embedding) / 4 FROM vec_chunks LIMIT 1",
                [],
                |r| r.get::<_, i64>(0),
            )
            .ok()
            .map(|d| d as usize)
        } else {
            None
        };

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            vec_enabled,
            vec_dim: Arc::new(Mutex::new(vec_dim)),
        })
    }

    /// Create the `vec0` virtual table once the embedding dimensionality is
    /// known (it is baked into the table definition). No-op if already created.
    fn ensure_vec_table(&self, conn: &Connection, dim: usize) -> Result<()> {
        let mut guard = self.vec_dim.lock();
        if guard.is_some() {
            return Ok(());
        }
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{dim}] distance_metric=cosine
            );"
        ))?;
        *guard = Some(dim);
        Ok(())
    }

    pub fn file_hash(&self, path: &str) -> Result<Option<String>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare("SELECT content_hash FROM files WHERE path=?1")?;
        let mut rows = stmt.query(params![path])?;
        Ok(rows.next()?.map(|r| r.get(0)).transpose()?)
    }

    /// Remove all derived data for a file. Called on change/delete before reindex.
    pub fn invalidate_file(&self, path: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE file=?1)",
            params![path],
        )?;
        if self.vec_enabled && self.vec_dim.lock().is_some() {
            conn.execute(
                "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE file=?1)",
                params![path],
            )?;
        }
        conn.execute("DELETE FROM chunks WHERE file=?1", params![path])?;
        conn.execute("DELETE FROM symbols WHERE file=?1", params![path])?;
        conn.execute("DELETE FROM edges WHERE src_symbol LIKE ?1 || '::%'", params![path])?;
        conn.execute("DELETE FROM files WHERE path=?1", params![path])?;
        Ok(())
    }

    pub fn upsert_file(
        &self,
        path: &str,
        hash: &str,
        size: i64,
        mtime_ms: i64,
        indexed_ms: i64,
    ) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO files(path,content_hash,size,mtime_ms,indexed_ms)
             VALUES(?1,?2,?3,?4,?5)
             ON CONFLICT(path) DO UPDATE SET
               content_hash=excluded.content_hash, size=excluded.size,
               mtime_ms=excluded.mtime_ms, indexed_ms=excluded.indexed_ms",
            params![path, hash, size, mtime_ms, indexed_ms],
        )?;
        Ok(())
    }

    pub fn insert_chunk(
        &self,
        file: &str,
        start: usize,
        end: usize,
        symbol: Option<&str>,
        content: &str,
    ) -> Result<i64> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO chunks(file,start_line,end_line,symbol,content)
             VALUES(?1,?2,?3,?4,?5)",
            params![file, start as i64, end as i64, symbol, content],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn insert_embedding(&self, chunk_id: i64, vec: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
        let conn = self.conn.lock();
        conn.execute(
            "INSERT OR REPLACE INTO embeddings(chunk_id,dim,vec) VALUES(?1,?2,?3)",
            params![chunk_id, vec.len() as i64, bytes],
        )?;
        // Mirror into the vec0 index for native KNN when available.
        if self.vec_enabled {
            self.ensure_vec_table(&conn, vec.len())?;
            conn.execute(
                "INSERT OR REPLACE INTO vec_chunks(chunk_id, embedding) VALUES(?1, ?2)",
                params![chunk_id, bytes],
            )?;
        }
        Ok(())
    }

    pub fn insert_symbol(
        &self, file: &str, name: &str, kind: &str, start: usize, end: usize,
    ) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO symbols(file,name,kind,start_line,end_line)
             VALUES(?1,?2,?3,?4,?5)",
            params![file, name, kind, start as i64, end as i64],
        )?;
        Ok(())
    }

    /// Insert a graph edge. `src` is namespaced as `<file>::<symbol>` (or
    /// `<file>::<file>` for file-level imports) so `invalidate_file` can prune a
    /// file's edges with a `src_symbol LIKE '<file>::%'` match.
    pub fn insert_edge(&self, src: &str, dst: &str, kind: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO edges(src_symbol,dst_symbol,kind) VALUES(?1,?2,?3)",
            params![src, dst, kind],
        )?;
        Ok(())
    }

    /// Top-k semantic search scoped to the files under `root`. Uses the
    /// sqlite-vec `vec0` index for native KNN when available (and an index
    /// exists), which scales past `ANN_SCALE_HINT` chunks; otherwise falls back
    /// to a brute-force cosine scan. Returned scores are cosine similarities in
    /// both paths.
    pub fn knn(&self, root: &str, query: &[f32], k: usize) -> Result<Vec<(ChunkRow, f32)>> {
        let prefix = root_prefix(root);
        if self.vec_enabled && self.vec_dim.lock().is_some() {
            // vec0 cannot filter by an arbitrary predicate inside the MATCH, so
            // over-fetch and filter by workspace afterwards.
            match self.knn_ann(query, k * 4) {
                Ok(rows) => {
                    let mut rows: Vec<_> = rows
                        .into_iter()
                        .filter(|(c, _)| c.file.starts_with(&prefix))
                        .collect();
                    rows.truncate(k);
                    return Ok(rows);
                }
                Err(e) => {
                    tracing::warn!("sqlite-vec knn failed, falling back to scan: {e}");
                }
            }
        }
        self.knn_bruteforce(&prefix, query, k)
    }

    /// Native KNN via the `vec0` virtual table. `distance_metric=cosine` means
    /// the reported distance is cosine distance, so similarity = 1 - distance.
    fn knn_ann(&self, query: &[f32], k: usize) -> Result<Vec<(ChunkRow, f32)>> {
        let qbytes: Vec<u8> = query.iter().flat_map(|f| f.to_le_bytes()).collect();
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT c.id,c.file,c.start_line,c.end_line,c.symbol,c.content, v.distance
             FROM vec_chunks v JOIN chunks c ON c.id = v.chunk_id
             WHERE v.embedding MATCH ?1 AND k = ?2
             ORDER BY v.distance",
        )?;
        let rows = stmt.query_map(params![qbytes, k as i64], |row| {
            let distance: f64 = row.get(6)?;
            Ok((
                ChunkRow {
                    id: row.get(0)?,
                    file: row.get(1)?,
                    start_line: row.get::<_, i64>(2)? as usize,
                    end_line: row.get::<_, i64>(3)? as usize,
                    symbol: row.get(4)?,
                    content: row.get(5)?,
                },
                (1.0 - distance) as f32,
            ))
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Brute-force cosine KNN over the chunks under `prefix`. Exact, but scans
    /// every matching embedding.
    fn knn_bruteforce(&self, prefix: &str, query: &[f32], k: usize) -> Result<Vec<(ChunkRow, f32)>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT c.id,c.file,c.start_line,c.end_line,c.symbol,c.content,e.vec
             FROM embeddings e JOIN chunks c ON c.id=e.chunk_id
             WHERE substr(c.file, 1, ?1) = ?2",
        )?;
        let mut scored: Vec<(ChunkRow, f32)> = Vec::new();
        let rows = stmt.query_map(params![prefix.len() as i64, prefix], |row| {
            let blob: Vec<u8> = row.get(6)?;
            let vec: Vec<f32> = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok((
                ChunkRow {
                    id: row.get(0)?,
                    file: row.get(1)?,
                    start_line: row.get::<_, i64>(2)? as usize,
                    end_line: row.get::<_, i64>(3)? as usize,
                    symbol: row.get(4)?,
                    content: row.get(5)?,
                },
                vec,
            ))
        })?;
        for r in rows {
            let (chunk, vec) = r?;
            scored.push((chunk, cosine(query, &vec)));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(k);
        Ok(scored)
    }

    /// Chunks defining the symbol `name` within the workspace at `root`.
    pub fn symbol_lookup(&self, root: &str, name: &str) -> Result<Vec<ChunkRow>> {
        let prefix = root_prefix(root);
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT c.id,c.file,c.start_line,c.end_line,c.symbol,c.content
             FROM symbols s JOIN chunks c
               ON c.file=s.file AND s.start_line BETWEEN c.start_line AND c.end_line
             WHERE s.name=?1 AND substr(s.file, 1, ?2) = ?3 LIMIT 20",
        )?;
        let rows = stmt.query_map(params![name, prefix.len() as i64, prefix], |row| {
            Ok(ChunkRow {
                id: row.get(0)?,
                file: row.get(1)?,
                start_line: row.get::<_, i64>(2)? as usize,
                end_line: row.get::<_, i64>(3)? as usize,
                symbol: row.get(4)?,
                content: row.get(5)?,
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Chunk containing lines `start..=end` of `file` (absolute path), used to
    /// pull in a definition reached through the code graph.
    pub fn chunk_at(&self, file: &str, line: usize) -> Result<Option<ChunkRow>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id,file,start_line,end_line,symbol,content FROM chunks
             WHERE file=?1 AND ?2 BETWEEN start_line AND end_line LIMIT 1",
        )?;
        let mut rows = stmt.query(params![file, line as i64])?;
        Ok(match rows.next()? {
            Some(row) => Some(ChunkRow {
                id: row.get(0)?,
                file: row.get(1)?,
                start_line: row.get::<_, i64>(2)? as usize,
                end_line: row.get::<_, i64>(3)? as usize,
                symbol: row.get(4)?,
                content: row.get(5)?,
            }),
            None => None,
        })
    }

    /// Names called from inside symbol `src` (`<file>::<symbol>`), plus the
    /// symbols that call `name` — one hop of the code graph in each direction.
    pub fn graph_neighbours(&self, root: &str, file: &str, name: &str) -> Result<(Vec<String>, Vec<String>)> {
        let prefix = root_prefix(root);
        let conn = self.conn.lock();
        let src = format!("{file}::{name}");
        let mut callees = Vec::new();
        {
            let mut stmt = conn.prepare(
                "SELECT DISTINCT dst_symbol FROM edges WHERE src_symbol=?1 AND kind='call' LIMIT 20",
            )?;
            for r in stmt.query_map(params![src], |r| r.get::<_, String>(0))? {
                if let Ok(c) = r { callees.push(c); }
            }
        }
        let mut callers = Vec::new();
        {
            let mut stmt = conn.prepare(
                "SELECT DISTINCT src_symbol FROM edges
                 WHERE dst_symbol=?1 AND kind='call' AND substr(src_symbol, 1, ?2) = ?3 LIMIT 20",
            )?;
            for r in stmt.query_map(params![name, prefix.len() as i64, prefix], |r| r.get::<_, String>(0))? {
                if let Ok(c) = r { callers.push(c); }
            }
        }
        Ok((callees, callers))
    }

    /// Every symbol definition under `root` as `(file, name, kind, start_line)`,
    /// ordered by file then line — the raw material for the project map.
    pub fn symbols_under(&self, root: &str) -> Result<Vec<(String, String, String, usize)>> {
        let prefix = root_prefix(root);
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT file,name,kind,start_line FROM symbols
             WHERE substr(file, 1, ?1) = ?2 ORDER BY file, start_line",
        )?;
        let rows = stmt.query_map(params![prefix.len() as i64, prefix], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get::<_, i64>(3)? as usize))
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Indexed file paths under `root`.
    pub fn files_under(&self, root: &str) -> Result<Vec<String>> {
        let prefix = root_prefix(root);
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT path FROM files WHERE substr(path, 1, ?1) = ?2 ORDER BY path",
        )?;
        let rows = stmt.query_map(params![prefix.len() as i64, prefix], |r| r.get::<_, String>(0))?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Files and chunks indexed under `root` (not the whole shared store).
    pub fn counts(&self, root: &str) -> Result<(usize, usize)> {
        let prefix = root_prefix(root);
        let conn = self.conn.lock();
        let files: i64 = conn.query_row(
            "SELECT COUNT(*) FROM files WHERE substr(path, 1, ?1) = ?2",
            params![prefix.len() as i64, prefix],
            |r| r.get(0),
        )?;
        let chunks: i64 = conn.query_row(
            "SELECT COUNT(*) FROM chunks WHERE substr(file, 1, ?1) = ?2",
            params![prefix.len() as i64, prefix],
            |r| r.get(0),
        )?;
        Ok((files as usize, chunks as usize))
    }

    /// Whole-store totals, for callers without a workspace.
    pub fn total_counts(&self) -> Result<(usize, usize)> {
        let conn = self.conn.lock();
        let files: i64 = conn.query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))?;
        let chunks: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
        Ok((files as usize, chunks as usize))
    }

    // ----- settings -------------------------------------------------------

    pub fn get_setting(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare("SELECT value FROM settings WHERE key=?1")?;
        let mut rows = stmt.query(params![key])?;
        Ok(rows.next()?.map(|r| r.get(0)).transpose()?)
    }

    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO settings(key,value) VALUES(?1,?2)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    // ----- conversation + memory -----------------------------------------

    pub fn add_message(&self, root: &str, role: &str, content: &str, now_ms: i64) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO messages(root,role,content,created_ms) VALUES(?1,?2,?3,?4)",
            params![root, role, content, now_ms],
        )?;
        Ok(())
    }

    /// The most recent `limit` turns for `root`, oldest first.
    pub fn recent_messages(&self, root: &str, limit: usize) -> Result<Vec<(String, String)>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT role,content FROM (
                 SELECT id,role,content FROM messages WHERE root=?1 ORDER BY id DESC LIMIT ?2
             ) ORDER BY id",
        )?;
        let rows = stmt.query_map(params![root, limit as i64], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    pub fn clear_messages(&self, root: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM messages WHERE root=?1", params![root])?;
        Ok(())
    }

    pub fn add_memory(&self, root: &str, note: &str, now_ms: i64) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO memory(root,note,created_ms) VALUES(?1,?2,?3)",
            params![root, note, now_ms],
        )?;
        Ok(())
    }

    /// The most recent `limit` memory notes for `root`, oldest first.
    pub fn recent_memory(&self, root: &str, limit: usize) -> Result<Vec<String>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT note FROM (
                 SELECT id,note FROM memory WHERE root=?1 ORDER BY id DESC LIMIT ?2
             ) ORDER BY id",
        )?;
        let rows = stmt.query_map(params![root, limit as i64], |r| r.get::<_, String>(0))?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Freeze a planned interface into the registry. Idempotent per `plan_id`:
    /// existing rows for the plan are cleared first. `INSERT OR IGNORE` enforces
    /// the `UNIQUE(plan_id,name)` constraint, so duplicate normalized names
    /// collapse to the first occurrence — the program, not the model, owns the
    /// final name set.
    pub fn freeze_plan(&self, plan_id: &str, symbols: &[PlanSymbol]) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction()?;
        tx.execute("DELETE FROM plan_symbols WHERE plan_id=?1", params![plan_id])?;
        for s in symbols {
            tx.execute(
                "INSERT OR IGNORE INTO plan_symbols(plan_id,module,name,kind,signature)
                 VALUES(?1,?2,?3,?4,?5)",
                params![plan_id, s.module, s.name, s.kind, s.signature],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Read back the frozen registry for a plan in insertion order. This is the
    /// canonical name set after normalization and dedup.
    pub fn get_plan_symbols(&self, plan_id: &str) -> Result<Vec<PlanSymbol>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT module,name,kind,signature FROM plan_symbols
             WHERE plan_id=?1 ORDER BY id",
        )?;
        let rows = stmt.query_map(params![plan_id], |row| {
            Ok(PlanSymbol {
                module: row.get(0)?,
                name: row.get(1)?,
                kind: row.get(2)?,
                signature: row.get(3)?,
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Seed one `pending` task per planned symbol. Idempotent: `INSERT OR IGNORE`
    /// against `UNIQUE(plan_id,name)` leaves existing tasks (and their progress)
    /// untouched on a re-run.
    pub fn create_tasks(&self, plan_id: &str, names: &[String]) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction()?;
        for name in names {
            tx.execute(
                "INSERT OR IGNORE INTO tasks(plan_id,name,status,attempts)
                 VALUES(?1,?2,'pending',0)",
                params![plan_id, name],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn pending_tasks(&self, plan_id: &str) -> Result<Vec<TaskRow>> {
        self.query_tasks(plan_id, Some("pending"))
    }

    pub fn get_tasks(&self, plan_id: &str) -> Result<Vec<TaskRow>> {
        self.query_tasks(plan_id, None)
    }

    fn query_tasks(&self, plan_id: &str, status: Option<&str>) -> Result<Vec<TaskRow>> {
        let conn = self.conn.lock();
        let sql = match status {
            Some(_) => "SELECT id,plan_id,name,status,code,attempts FROM tasks
                        WHERE plan_id=?1 AND status=?2 ORDER BY id",
            None => "SELECT id,plan_id,name,status,code,attempts FROM tasks
                     WHERE plan_id=?1 ORDER BY id",
        };
        let mut stmt = conn.prepare(sql)?;
        let map = |row: &rusqlite::Row| {
            Ok(TaskRow {
                id: row.get(0)?,
                plan_id: row.get(1)?,
                name: row.get(2)?,
                status: row.get(3)?,
                code: row.get(4)?,
                attempts: row.get(5)?,
            })
        };
        let rows = match status {
            Some(s) => stmt.query_map(params![plan_id, s], map)?.collect::<Vec<_>>(),
            None => stmt.query_map(params![plan_id], map)?.collect::<Vec<_>>(),
        };
        Ok(rows.into_iter().filter_map(|r| r.ok()).collect())
    }

    pub fn set_task_status(&self, id: i64, status: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("UPDATE tasks SET status=?2 WHERE id=?1", params![id, status])?;
        Ok(())
    }

    pub fn save_task_code(&self, id: i64, code: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("UPDATE tasks SET code=?2 WHERE id=?1", params![id, code])?;
        Ok(())
    }

    pub fn bump_attempt(&self, id: i64) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("UPDATE tasks SET attempts=attempts+1 WHERE id=?1", params![id])?;
        Ok(())
    }

    /// Append a checker failure. The error log is never overwritten, so each
    /// retry can be shown every prior attempt and exactly why it failed.
    pub fn log_error(&self, task_id: i64, attempt: i64, checker: &str, message: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO errors(task_id,attempt,checker,message) VALUES(?1,?2,?3,?4)",
            params![task_id, attempt, checker, message],
        )?;
        Ok(())
    }

    /// Full failure history for a task as (attempt, checker, message), oldest
    /// first — the correction signal fed back into the next generation.
    pub fn get_error_history(&self, task_id: i64) -> Result<Vec<(i64, String, String)>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT attempt,checker,message FROM errors WHERE task_id=?1 ORDER BY id",
        )?;
        let rows = stmt.query_map(params![task_id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }
}

/// `root` normalised to a path prefix with exactly one trailing separator, so
/// `substr(path, 1, len) = prefix` selects precisely the files under it.
pub fn root_prefix(root: &str) -> String {
    format!("{}/", root.trim_end_matches('/'))
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return -1.0;
    }
    let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na.sqrt() * nb.sqrt()) }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> Store {
        let dir = std::env::temp_dir().join(format!("codebot-store-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        Store::open(&dir.join("t.db")).unwrap()
    }

    #[test]
    fn file_row_then_chunks_then_reindex() {
        let s = temp_store();
        let root = "/ws/proj";
        let f = "/ws/proj/src/a.rs";
        s.upsert_file(f, "", 1, 1, 1).unwrap();
        let id = s.insert_chunk(f, 1, 2, Some("a"), "fn a() {}").unwrap();
        s.insert_embedding(id, &[0.1, 0.2]).unwrap();
        s.upsert_file(f, "h1", 1, 1, 1).unwrap();
        assert_eq!(s.counts(root).unwrap(), (1, 1));
        assert_eq!(s.counts("/ws/other").unwrap(), (0, 0));
        assert_eq!(s.files_under(root).unwrap(), vec![f.to_string()]);

        // Re-index path: invalidate, then rebuild.
        s.invalidate_file(f).unwrap();
        assert_eq!(s.counts(root).unwrap(), (0, 0));
        s.upsert_file(f, "", 1, 1, 1).unwrap();
        s.insert_chunk(f, 1, 2, None, "x").unwrap();
        assert_eq!(s.counts(root).unwrap(), (1, 1));
    }

    #[test]
    fn settings_messages_memory_roundtrip() {
        let s = temp_store();
        assert_eq!(s.get_setting("k").unwrap(), None);
        s.set_setting("k", "v").unwrap();
        assert_eq!(s.get_setting("k").unwrap().as_deref(), Some("v"));
        s.add_message("/r", "user", "hi", 1).unwrap();
        s.add_message("/r", "assistant", "yo", 2).unwrap();
        s.add_message("/other", "user", "no", 3).unwrap();
        let m = s.recent_messages("/r", 10).unwrap();
        assert_eq!(m, vec![("user".to_string(), "hi".to_string()), ("assistant".to_string(), "yo".to_string())]);
        s.add_memory("/r", "note", 4).unwrap();
        assert_eq!(s.recent_memory("/r", 5).unwrap(), vec!["note".to_string()]);
        s.clear_messages("/r").unwrap();
        assert!(s.recent_messages("/r", 10).unwrap().is_empty());
    }
}
