import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
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

class MyHomePage extends StatelessWidget {
  const MyHomePage({Key? key, required this.title}) : super(key: key);
  final String title;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            TextButton(
                style: defaultButtonStyle(),
                onPressed: () {
                  Navigator.push(context, MaterialPageRoute(builder: (context) {
                    return const ScanningPage(title: "Scanner Environment");
                  }));
                },
                child: Text('Start Scan')),
            SizedBox(height: 15),
            TextButton(
                style: defaultButtonStyle(),
                onPressed: () {
                  Navigator.push(context, MaterialPageRoute(builder: (context) {
                    return const ViewModelPage(title: "3D Render Viewer");
                  }));
                },
                child: Text('View 3D Render'))
          ],
        ),
      ),
    );
  }
}

class ScanningPage extends StatelessWidget {
  const ScanningPage({Key? key, required this.title}) : super(key: key);

  final String title;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          children: [
            Text("Scanning Environment goes here!"),
            SizedBox(height: 15),
            TextButton(onPressed: () {
              Navigator.pop(context)
            }), child: child)

          ],
        ),
      ),
    );
  }
}

class ViewModelPage extends StatelessWidget {
  const ViewModelPage({Key? key, required this.title}) : super(key: key);

  final String title;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Text("3D Render goes here!"),
      ),
    );
  }
}

ButtonStyle defaultButtonStyle() => ButtonStyle(
      backgroundColor: MaterialStateProperty.all<Color>(Colors.purpleAccent),
      foregroundColor: MaterialStateProperty.all<Color>(Colors.white),
      padding: MaterialStateProperty.all<EdgeInsets>(EdgeInsets.all(25)),
    );
