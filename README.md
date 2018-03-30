# Backpropagation Neural Network

Small neural network I created after a few tutorials. It can have n amount layers and nodes per layer. It has tanh, sigmoid and relu as activation and output functions. Success rate with the MINST training data is around 96 to 97 percent. EMINST hovers around 75 to 80, but I am sure it could be optimized to get more.

Also hosted on very simple python REST Heroku server for testing with Android app. It expects a black on white image of a character. Server removes transparent background, crops image to a quadratic format and resizes it. Also all values get normalized/scaled for the network. The POST URL is: https://char-recognization.herokuapp.com/

Android app allows the user to draw a single character and let's the REST API guess what character it is. Download the debug APK here: https://www.file-upload.net/download-13058921/app-debug.apk.html
  
Screenshot:

<img width="200" src="https://i.imgur.com/Hk4QCaR.png">

Processed image on server (trimmed whitespace, inverted, scaled to 20x20 box, put in 28x28 background):

<img width="200" src="https://i.imgur.com/XPtjHDP.png">

Good materials:

<ul>
  <li>https://www.amazon.de/Make-Your-Own-Neural-Network/dp/1530826608/ref=sr_1_1?ie=UTF8&qid=1521930572&sr=8-1&keywords=make+your+own+neural+network</li>
  <li>https://www.youtube.com/watch?v=aircAruvnKk</li>
  <li>https://www.youtube.com/playlist?list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh</li>
</ul>  
