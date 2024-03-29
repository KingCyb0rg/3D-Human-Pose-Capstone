import 'package:flutter/material.dart';

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
                    image: AssetImage("assets/images/tylmen_splash.jpg"))),
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
              child: Container(
                alignment: Alignment.center,
                child: Text('OpenCV enviroment goes '),
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

class ViewModelPage extends StatefulWidget {
  const ViewModelPage({Key? key, required this.title}) : super(key: key);

  final String title;

  @override
  _ViewModelPageState createState() => _ViewModelPageState();
}

class _ViewModelPageState extends State<ViewModelPage> {
  @override
  

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
