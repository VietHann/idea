import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'app/mori_app.dart';
import 'core/storage/local_database.dart';
import 'core/theme/mori_theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  FlutterError.onError = (details) {
    FlutterError.presentError(details);
  };
  PlatformDispatcher.instance.onError = (error, stack) {
    debugPrint('MORI uncaught error: $error\n$stack');
    return true;
  };

  runApp(const MoriBootstrap());
}

class MoriBootstrap extends StatefulWidget {
  const MoriBootstrap({super.key});

  @override
  State<MoriBootstrap> createState() => _MoriBootstrapState();
}

class _MoriBootstrapState extends State<MoriBootstrap> {
  late Future<LocalDatabase> _database;

  @override
  void initState() {
    super.initState();
    _database = LocalDatabase.open();
  }

  void _retry() {
    setState(() => _database = LocalDatabase.open());
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<LocalDatabase>(
      future: _database,
      builder: (context, snapshot) {
        if (snapshot.hasData) {
          return ProviderScope(
            overrides: [
              localDatabaseProvider.overrideWithValue(snapshot.data!),
            ],
            child: const MoriApp(),
          );
        }

        if (snapshot.hasError) {
          return MaterialApp(
            debugShowCheckedModeBanner: false,
            theme: MoriTheme.light,
            home: Scaffold(
              body: SafeArea(
                child: Center(
                  child: Padding(
                    padding: const EdgeInsets.all(28),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(
                          Icons.cloud_off_rounded,
                          size: 42,
                          color: MoriColors.rose,
                        ),
                        const SizedBox(height: 18),
                        Text(
                          'MORI could not open its local library.',
                          textAlign: TextAlign.center,
                          style: Theme.of(context).textTheme.titleLarge,
                        ),
                        const SizedBox(height: 10),
                        const Text(
                          'Your data stays on this iPhone. Retry opening the local database.',
                          textAlign: TextAlign.center,
                          style: TextStyle(color: MoriColors.muted),
                        ),
                        const SizedBox(height: 22),
                        FilledButton.icon(
                          onPressed: _retry,
                          icon: const Icon(Icons.refresh_rounded),
                          label: const Text('Try again'),
                        ),
                        const SizedBox(height: 14),
                        Text(
                          snapshot.error.toString(),
                          textAlign: TextAlign.center,
                          maxLines: 4,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            color: MoriColors.muted,
                            fontSize: 11,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          );
        }

        return MaterialApp(
          debugShowCheckedModeBanner: false,
          theme: MoriTheme.light,
          home: const Scaffold(
            body: Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    'MORI',
                    style: TextStyle(
                      fontSize: 26,
                      fontWeight: FontWeight.w900,
                      letterSpacing: 4,
                      color: MoriColors.ink,
                    ),
                  ),
                  SizedBox(height: 16),
                  SizedBox(
                    width: 22,
                    height: 22,
                    child: CircularProgressIndicator(strokeWidth: 2.2),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}
