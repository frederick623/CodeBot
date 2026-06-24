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
            CREATE INDEX IF NOT EXISTS idx_plan_symbols_plan ON plan_symbols(plan_id);
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

    /// Top-k semantic search. Uses the sqlite-vec `vec0` index for native KNN
    /// when available (and an index exists), which scales past `ANN_SCALE_HINT`
    /// chunks; otherwise falls back to a brute-force cosine scan. Returned scores
    /// are cosine similarities in both paths.
    pub fn knn(&self, query: &[f32], k: usize) -> Result<Vec<(ChunkRow, f32)>> {
        if self.vec_enabled && self.vec_dim.lock().is_some() {
            match self.knn_ann(query, k) {
                Ok(rows) => return Ok(rows),
                Err(e) => {
                    tracing::warn!("sqlite-vec knn failed, falling back to scan: {e}");
                }
            }
        }
        self.knn_bruteforce(query, k)
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

    /// Brute-force cosine KNN. Exact, but scans every embedding.
    fn knn_bruteforce(&self, query: &[f32], k: usize) -> Result<Vec<(ChunkRow, f32)>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT c.id,c.file,c.start_line,c.end_line,c.symbol,c.content,e.vec
             FROM embeddings e JOIN chunks c ON c.id=e.chunk_id",
        )?;
        let mut scored: Vec<(ChunkRow, f32)> = Vec::new();
        let rows = stmt.query_map([], |row| {
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

    pub fn symbol_lookup(&self, name: &str) -> Result<Vec<ChunkRow>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT c.id,c.file,c.start_line,c.end_line,c.symbol,c.content
             FROM symbols s JOIN chunks c
               ON c.file=s.file AND s.start_line BETWEEN c.start_line AND c.end_line
             WHERE s.name=?1 LIMIT 20",
        )?;
        let rows = stmt.query_map(params![name], |row| {
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

    pub fn counts(&self) -> Result<(usize, usize)> {
        let conn = self.conn.lock();
        let files: i64 = conn.query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))?;
        let chunks: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
        Ok((files as usize, chunks as usize))
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