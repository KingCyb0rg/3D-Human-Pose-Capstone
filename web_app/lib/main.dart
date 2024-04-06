import 'dart:collection';
import 'package:flutter/material.dart';
import 'package:flutter_inappwebview/flutter_inappwebview.dart';
import 'dart:js' as js;

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
        primarySwatch: Colors.purple,
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
    return Scaffold(
      body: Stack(
        children: <Widget>[
          Container(
            decoration: BoxDecoration(
                image: DecorationImage(
                    image: AssetImage("images/tylmen_splash.jpg"))),
          ),
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                TextButton(
                    style: defaultButtonStyle(),
                    onPressed: () {
                      Navigator.push(context,
                          MaterialPageRoute(builder: (context) {
                        return const ScanningPage(title: "Scanner Environment");
                      }));
                    },
                    child: Text('Start Scan')),
                SizedBox(height: 15),
                TextButton(
                    style: defaultButtonStyle(),
                    onPressed: () {
                      Navigator.push(context,
                          MaterialPageRoute(builder: (context) {
                        return const ViewModelPage(title: "3D Render Viewer");
                      }));
                    },
                    child: Text('View 3D Render'))
              ],
            ),
          ),
        ],
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
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((timeStamp) async {
      await showDialog<String>(
          context: context,
          builder: (BuildContext context) => new AlertDialog(
                title: new Text("Scanning Setup"),
                content: new Text(
                    "Please confirm that you are outdoors before starting the scan."),
                backgroundColor: Colors.grey,
                actions: <Widget>[
                  new TextButton(
                      onPressed: () {
                        Navigator.of(context).pop();
                      },
                      child: new Text("Yes")),
                ],
              ));
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Container(
                alignment: Alignment.topLeft,
                padding: EdgeInsets.all(10),
                clipBehavior: Clip.hardEdge,
                decoration: defaultBoxDecoration(),
                child: BackButton(
                  onPressed: () {
                    Navigator.pop(context);
                  },
                ),
              ),
              // AR environment placeholder
              Expanded(
                child: Center(
                  child: Container(
                    padding: EdgeInsets.all(5),
                    width: double.infinity,
                    height: double.infinity,
                    child: FutureBuilder<InAppWebView>(
                      future: callAsyncWebView(),
                      builder: (context, AsyncSnapshot<InAppWebView> snapshot) {
                        if (snapshot.hasData)
                          return snapshot.requireData;
                        else
                          return LinearProgressIndicator();
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
                child: ElevatedButton(
                  child: const Text('Start'),
                  onPressed: () {
                    setState(() {
                      js.context.callMethod('startCapture');
                    });
                  },
                  style: defaultButtonStyle(),
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
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Container(
              alignment: Alignment.topLeft,
              padding: EdgeInsets.all(10),
              clipBehavior: Clip.hardEdge,
              decoration: defaultBoxDecoration(),
              child: BackButton(
                onPressed: () {
                  Navigator.pop(context);
                },
              ),
            ),
            // Model viewer environment placeholder
            Expanded(
              child: Container(
                alignment: Alignment.center,
                child: Text('Model view goes HERE!!!!'),
              ),
            ),
            Container(
              alignment: Alignment.center,
              padding: EdgeInsets.all(10),
              clipBehavior: Clip.hardEdge,
              decoration: defaultBoxDecoration(),
              child: ElevatedButton(
                child: const Text('Start'),
                onPressed: () {},
                style: defaultButtonStyle(),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

Future<InAppWebView> callAsyncWebView() async =>
    await Future.delayed(Duration(seconds: 2), () => createWebView());

InAppWebView createWebView() {
  return InAppWebView(
    initialFile: "web/opencv.html",
    initialUserScripts: UnmodifiableListView<UserScript>(
      [
        UserScript(
          source: 'web/scannerapp.js',
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

ButtonStyle defaultButtonStyle() => ButtonStyle(
      backgroundColor: MaterialStateProperty.all<Color>(Colors.purpleAccent),
      foregroundColor: MaterialStateProperty.all<Color>(Colors.white),
      padding: MaterialStateProperty.all<EdgeInsets>(EdgeInsets.all(25)),
    );

BoxDecoration defaultBoxDecoration() => BoxDecoration(
    color: Colors.purple,
    border: Border.all(
      color: Colors.black,
      style: BorderStyle.solid,
    ));
