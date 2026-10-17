import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';

final localDatabaseProvider = Provider<LocalDatabase>((ref) {
  throw StateError('LocalDatabase must be overridden after bootstrap.');
});

class LocalDatabase {
  LocalDatabase._(this.db);

  final Database db;
  static const _name = 'mori.sqlite';
  static const schemaVersion = 5;

  static Future<LocalDatabase> open() async {
    final root = await getDatabasesPath();
    final path = p.join(root, _name);

    final database = await openDatabase(
      path,
      version: schemaVersion,
      onConfigure: (db) async {
        await db.execute('PRAGMA foreign_keys = ON');
      },
      onCreate: (db, version) async => _create(db),
      onUpgrade: (db, oldVersion, newVersion) async {
        await _ensureSchema(db);
      },
      onOpen: (db) async {
        await _ensureSchema(db);
      },
    );

    return LocalDatabase._(database);
  }

  static Future<void> _create(Database db) async {
    await db.execute('''
      CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL,
        captured_at INTEGER NOT NULL,
        day_key TEXT NOT NULL,
        caption TEXT NOT NULL DEFAULT '',
        mood TEXT NOT NULL,
        image_path TEXT NOT NULL,
        audio_path TEXT,
        audio_duration_ms INTEGER NOT NULL DEFAULT 0,
        latitude REAL,
        longitude REAL,
        place_name TEXT,
        is_favorite INTEGER NOT NULL DEFAULT 0 CHECK(is_favorite IN (0,1)),
        is_daily INTEGER NOT NULL DEFAULT 1 CHECK(is_daily IN (0,1))
      )
    ''');
    await _ensureMemoryColumns(db);
    await _createIndexes(db);
    await _createThoughts(db);
    await _createMeta(db);
  }

  static Future<void> _ensureSchema(Database db) async {
    final tables = await db.rawQuery(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'",
    );

    if (tables.isEmpty) {
      await _create(db);
      return;
    }

    await _ensureMemoryColumns(db);
    await _createIndexes(db);
    await _createThoughts(db);
    await _ensureThoughtColumns(db);
    await _createMeta(db);
  }

  static Future<Set<String>> _columns(Database db, String table) async {
    final rows = await db.rawQuery('PRAGMA table_info($table)');
    return rows.map((row) => row['name'] as String).toSet();
  }

  static Future<void> _ensureMemoryColumns(Database db) async {
    final columns = await _columns(db, 'memories');
    final additions = <String, String>{
      'created_at': 'INTEGER NOT NULL DEFAULT 0',
      'captured_at': 'INTEGER NOT NULL DEFAULT 0',
      'day_key': "TEXT NOT NULL DEFAULT ''",
      'caption': "TEXT NOT NULL DEFAULT ''",
      'mood': "TEXT NOT NULL DEFAULT 'calm'",
      'image_path': "TEXT NOT NULL DEFAULT ''",
      'place_name': 'TEXT',
      'audio_path': 'TEXT',
      'audio_duration_ms': 'INTEGER NOT NULL DEFAULT 0',
      'latitude': 'REAL',
      'longitude': 'REAL',
      'is_favorite': 'INTEGER NOT NULL DEFAULT 0',
      'is_daily': 'INTEGER NOT NULL DEFAULT 1',
    };

    for (final entry in additions.entries) {
      if (!columns.contains(entry.key)) {
        await db.execute(
          'ALTER TABLE memories ADD COLUMN ${entry.key} ${entry.value}',
        );
      }
    }
  }

  static Future<void> _createIndexes(Database db) async {
    await db.execute(
      'CREATE INDEX IF NOT EXISTS idx_memories_captured_at ON memories(captured_at DESC)',
    );
    await db.execute(
      'CREATE INDEX IF NOT EXISTS idx_memories_day_key ON memories(day_key)',
    );
  }

  static Future<void> _createThoughts(Database db) async {
    await db.execute('''
      CREATE TABLE IF NOT EXISTS thoughts (
        id TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        body TEXT NOT NULL,
        cluster TEXT NOT NULL DEFAULT 'Life',
        color_key TEXT NOT NULL DEFAULT 'blush',
        related_memory_id TEXT,
        is_pinned INTEGER NOT NULL DEFAULT 0 CHECK(is_pinned IN (0,1)),
        FOREIGN KEY(related_memory_id) REFERENCES memories(id) ON DELETE SET NULL
      )
    ''');
    await db.execute(
      'CREATE INDEX IF NOT EXISTS idx_thoughts_updated_at ON thoughts(updated_at DESC)',
    );
    await db.execute(
      'CREATE INDEX IF NOT EXISTS idx_thoughts_cluster ON thoughts(cluster)',
    );
  }

  static Future<void> _ensureThoughtColumns(Database db) async {
    final columns = await _columns(db, 'thoughts');
    final additions = <String, String>{
      'created_at': 'INTEGER NOT NULL DEFAULT 0',
      'updated_at': 'INTEGER NOT NULL DEFAULT 0',
      'title': "TEXT NOT NULL DEFAULT ''",
      'body': "TEXT NOT NULL DEFAULT ''",
      'cluster': "TEXT NOT NULL DEFAULT 'Life'",
      'color_key': "TEXT NOT NULL DEFAULT 'blush'",
      'related_memory_id': 'TEXT',
      'is_pinned': 'INTEGER NOT NULL DEFAULT 0',
    };

    for (final entry in additions.entries) {
      if (!columns.contains(entry.key)) {
        await db.execute(
          'ALTER TABLE thoughts ADD COLUMN ${entry.key} ${entry.value}',
        );
      }
    }
  }

  static Future<void> _createMeta(Database db) async {
    await db.execute('''
      CREATE TABLE IF NOT EXISTS app_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      )
    ''');
  }
}
