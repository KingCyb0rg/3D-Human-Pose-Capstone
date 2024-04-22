import 'dart:collection';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_inappwebview/flutter_inappwebview.dart';
import 'package:video_player/video_player.dart';
import 'package:flutter_cube/flutter_cube.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  // This widget is the root of your application.
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '3D Model Web Demo',
      theme: ThemeData(
        useMaterial3: true,
        primarySwatch: Colors.purple,
        textButtonTheme: TextButtonThemeData(
            style: ButtonStyle(
          backgroundColor:
              MaterialStateProperty.all<Color>(Colors.purpleAccent),
          foregroundColor: MaterialStateProperty.all<Color>(Colors.white),
          padding: MaterialStateProperty.all<EdgeInsets>(EdgeInsets.all(25)),
        )),
      ),
      home: MyHomePage(title: '3D Model Web App Demo'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, required this.title}) : super(key: key);
  final String title;

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: <Widget>[
          Container(
            child: new Text(
              "3D Model Creator Demo",
              style: Theme.of(context).textTheme.headlineLarge,
            ),
          ),
          SizedBox(height: 50),
          new Container(
              decoration: BoxDecoration(
                  image: DecorationImage(
                      image: AssetImage("images/tylmen_splash.jpg")))),
          TextButton(
              style: Theme.of(context).textButtonTheme.style,
              onPressed: () {
                Navigator.push(context, MaterialPageRoute(builder: (context) {
                  return const VideoPage(title: "Scan Instructions");
                }));
              },
              child: Text('Start Scan')),
          SizedBox(
            height: 10,
          ),
          TextButton(
              style: Theme.of(context).textButtonTheme.style,
              onPressed: () {
                Navigator.push(context, MaterialPageRoute(builder: (context) {
                  return const ViewModelPage(title: "3D Render Viewer");
                }));
              },
              child: Text('View 3D Render')),
          SizedBox(
            height: 10,
          ),
          TextButton(
              style: Theme.of(context).textButtonTheme.style,
              onPressed: () {
                SystemNavigator.pop();
              },
              child: new Text("Exit")),
        ],
      ),
    );
  }
}

class VideoPage extends StatefulWidget {
  const VideoPage({Key? key, required this.title}) : super(key: key);
  final String title;

  @override
  _VideoPageState createState() => _VideoPageState();
}

class _VideoPageState extends State<VideoPage> {
  late VideoPlayerController _playerController;

  @override
  void initState() {
    super.initState();
    _playerController = VideoPlayerController.asset("videos/instructions.mp4")
      ..initialize().then((_) => {setState(() {})});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            Container(
              alignment: Alignment.topLeft,
              padding: EdgeInsets.all(10),
              clipBehavior: Clip.hardEdge,
              decoration: defaultBoxDecoration(),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  BackButton(
                    onPressed: () {
                      Navigator.pop(context);
                    },
                    color: Colors.white,
                  )
                ],
              ),
            ),
            Expanded(
              child: _playerController.value.isInitialized
                  ? AspectRatio(
                      aspectRatio: _playerController.value.aspectRatio,
                      child: VideoPlayer(_playerController),
                    )
                  : Container(),
            ),
            Container(
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  // Rewind Button
                  IconButton(
                    onPressed: () {
                      setState(() {
                        _playerController.seekTo(
                            _playerController.value.position -
                                const Duration(seconds: 10));
                      });
                    },
                    icon: Icon(Icons.arrow_back),
                  ),
                  SizedBox(
                    width: 10,
                  ),
                  // Play Pause Button
                  IconButton(
                    onPressed: () {
                      setState(() {
                        _playerController.value.isPlaying
                            ? _playerController.pause()
                            : _playerController.play();
                      });
                    },
                    icon: Icon(_playerController.value.isPlaying
                        ? Icons.pause
                        : Icons.play_arrow),
                  ),
                  SizedBox(
                    width: 10,
                  ),
                  // Fast Forward Button
                  IconButton(
                    onPressed: () {
                      setState(() {
                        _playerController.seekTo(
                            _playerController.value.position +
                                const Duration(seconds: 10));
                      });
                    },
                    icon: Icon(Icons.arrow_forward),
                  ),
                  SizedBox(width: 50),
                  // Next Page Button
                  FloatingActionButton(
                      onPressed: () {
                        Navigator.push(context,
                            MaterialPageRoute(builder: (context) {
                          return const ScanningPage(title: "Scanning Page");
                        }));
                      },
                      child: Icon(Icons.forward)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class ScanningPage extends StatefulWidget {
  const ScanningPage({Key? key, required this.title}) : super(key: key);

  final String title;

  @override
  _ScanningPageState createState() => _ScanningPageState();
}

class _ScanningPageState extends State<ScanningPage> {
  InAppWebView createWebView() {
    return InAppWebView(
      onWebViewCreated: (controller) {},
      initialFile: "web/ScannerApp.html",
      initialUserScripts: UnmodifiableListView<UserScript>(
        [
          UserScript(
            source: 'web/ScannerApp.js',
            injectionTime: UserScriptInjectionTime.AT_DOCUMENT_END,
          )
        ],
      ),
      initialSettings: InAppWebViewSettings(
        javaScriptEnabled: true,
        useOnLoadResource: false,
        verticalScrollBarEnabled: false,
        horizontalScrollBarEnabled: false,
        disableHorizontalScroll: true,
        disableVerticalScroll: true,
        useWideViewPort: false,
      ),
    );
  }

  Future<InAppWebView> callAsyncWebView() async =>
      await Future.delayed(Duration(seconds: 3), () => createWebView());

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((timeStamp) async {
      await showDialog<String>(
          context: context,
          builder: (BuildContext context) {
            var data = """
Please make sure that you are outside before starting the scanning process.
The background will turn green when you assume the correct pose. 
Click the 'Start Recording" Button once you are ready to start the scanning process.
Recording can be stopped at any time using the "Stop Recording" Button.
                    """;
            return new AlertDialog(
              title: new Text("Scanning Setup"),
              content: new Text(data),
              backgroundColor: Colors.white,
              actions: <Widget>[
                new TextButton(
                    onPressed: () {
                      Navigator.of(context).pop();
                    },
                    child: new Text("Okay")),
              ],
            );
          });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey,
      body: Center(
        child: Column(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Container(
                alignment: Alignment.topLeft,
                padding: EdgeInsets.all(10),
                clipBehavior: Clip.hardEdge,
                decoration: defaultBoxDecoration(),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    BackButton(
                      onPressed: () {
                        Navigator.pop(context);
                      },
                      color: Colors.white,
                    ),
                    IconButton(
                      onPressed: () {},
                      icon: new Icon(Icons.help_outline),
                      color: Colors.white,
                    ),
                  ],
                ),
              ),
              Expanded(
                child: Center(
                  child: SizedBox(
                    width: double.infinity,
                    height: double.infinity,
                    child: FutureBuilder<InAppWebView>(
                      future: callAsyncWebView(),
                      builder: (context, AsyncSnapshot<InAppWebView> snapshot) {
                        if (snapshot.hasData) {
                          return snapshot.requireData;
                        } else
                          return Center(
                            child: SizedBox(
                                width: 50,
                                height: 50,
                                child: CircularProgressIndicator(
                                  valueColor: AlwaysStoppedAnimation(
                                      Colors.purpleAccent),
                                  strokeWidth: 5,
                                )),
                          );
                      },
                    ),
                  ),
                ),
              ),
              Container(
                alignment: Alignment.center,
                padding: EdgeInsets.all(10),
                clipBehavior: Clip.hardEdge,
                decoration: defaultBoxDecoration(),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [],
                ),
              ),
            ]),
      ),
    );
  }
}

class ViewModelPage extends StatefulWidget {
  const ViewModelPage({Key? key, required this.title}) : super(key: key);

  final String title;

  @override
  _ViewModelPageState createState() => _ViewModelPageState();
}

class _ViewModelPageState extends State<ViewModelPage> {
  late Object model;

  // @override
  // void initState() {
  //   model =
  // }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Container(
              alignment: Alignment.topLeft,
              clipBehavior: Clip.hardEdge,
              decoration: defaultBoxDecoration(),
              child: BackButton(
                  onPressed: () {
                    Navigator.pop(context);
                  },
                  color: Colors.white),
            ),
            // Model viewer environment placeholder
            Expanded(
              child: Container(),
              // child: Cube(),
            ),
            Container(
              alignment: Alignment.center,
              padding: EdgeInsets.all(10),
              clipBehavior: Clip.hardEdge,
              decoration: defaultBoxDecoration(),
              child: ElevatedButton(
                child: const Text('Start'),
                onPressed: () async {
                  await showDialog<String>(
                      context: context,
                      builder: (BuildContext context) => new AlertDialog(
                            title: new Text("Work In Progress View"),
                            icon: Icon(Icons.warning, color: Colors.black),
                            content: new Text(
                                "Model Viewer is currently under development! Thank you for your patience!"),
                            actions: <Widget>[
                              TextButton(
                                  onPressed: () {
                                    Navigator.pop(context);
                                  },
                                  child: new Text("OK"))
                            ],
                          ));
                },
                style: Theme.of(context).textButtonTheme.style,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

BoxDecoration defaultBoxDecoration() => BoxDecoration(
    color: Colors.purple,
    border: Border.all(
      color: Colors.black,
      style: BorderStyle.solid,
    ));
