# Run from: C:\Users\NAVEENKUMAR\stress_detection\
# Command:  python write_live.py

content = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Live Detection - StressNet</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:\'Segoe UI\',sans-serif;background:linear-gradient(135deg,#0d0d2b,#1a1a4e);min-height:100vh;color:#e2e8f0;padding:20px;}
.header{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;padding-bottom:14px;border-bottom:1px solid rgba(255,255,255,0.08);}
.brand{font-size:20px;font-weight:800;background:linear-gradient(90deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.grid{display:grid;grid-template-columns:1fr 320px;gap:16px;}
.card{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:16px;}
.card-title{font-size:12px;font-weight:600;color:#a0aec0;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;}
.btn-start{padding:9px 20px;background:linear-gradient(135deg,#68d391,#38a169);border:none;border-radius:8px;color:#fff;font-size:13px;font-weight:700;cursor:pointer;}
.btn-stop{padding:9px 20px;background:rgba(252,73,73,0.2);border:1px solid rgba(252,73,73,0.4);border-radius:8px;color:#fc8181;font-size:13px;font-weight:700;cursor:pointer;display:none;}
.video-wrap{position:relative;width:100%;border-radius:10px;overflow:hidden;background:#000;min-height:360px;}
video{width:100%;display:block;border-radius:10px;}
canvas#overlay{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;}
.no-cam{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;color:#4a5568;}
.status-bar{display:flex;justify-content:space-between;margin-top:8px;font-size:11px;color:#718096;}
.nav-link{padding:7px 14px;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:8px;color:#a0aec0;text-decoration:none;font-size:12px;}
</style>
</head>
<body>
<div class="header">
  <div style="display:flex;align-items:center;gap:12px;">
    <span style="font-size:24px;">&#129504;</span>
    <span class="brand">StressNet</span>
    <span style="font-size:13px;color:#718096;">Live Detection</span>
  </div>
  <div style="display:flex;gap:8px;">
    <a href="/dashboard/" class="nav-link">&#128202; Dashboard</a>
    <a href="/history/"   class="nav-link">&#128203; History</a>
    <a href="/logout/"    class="nav-link">&#128682; Logout</a>
  </div>
</div>

<div class="grid">
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
      <span class="card-title" style="margin-bottom:0;">Camera Feed</span>
      <div style="display:flex;gap:8px;">
        <button class="btn-start" id="startBtn" onclick="startDetection()">&#9654; Start</button>
        <button class="btn-stop"  id="stopBtn"  onclick="stopDetection()">&#9632; Stop</button>
      </div>
    </div>
    <div class="video-wrap">
      <video id="video" autoplay playsinline muted></video>
      <canvas id="overlay"></canvas>
      <div class="no-cam" id="noCamera">
        <div style="font-size:48px;">&#128247;</div>
        <div style="font-size:14px;margin-top:8px;">Click Start to begin</div>
      </div>
    </div>
    <div class="status-bar">
      <span id="statusText">Status: Stopped</span>
      <span id="fpsText">FPS: --</span>
      <span id="faceText">Faces: 0</span>
    </div>
  </div>

  <div style="display:flex;flex-direction:column;gap:12px;">
    <div class="card">
      <div class="card-title">Current Detection</div>
      <div style="text-align:center;padding:16px 0;">
        <div id="stressText" style="font-size:32px;font-weight:800;color:#4a5568;">--</div>
        <div id="emotionText" style="font-size:16px;color:#718096;text-transform:capitalize;margin-top:4px;">No detection yet</div>
      </div>
      <div style="height:6px;background:rgba(255,255,255,0.08);border-radius:3px;margin:10px 0 4px;">
        <div id="confFill" style="height:100%;border-radius:3px;background:#4a5568;width:0%;transition:width 0.4s;"></div>
      </div>
      <div id="confText" style="font-size:11px;color:#718096;text-align:center;">Confidence: 0%</div>
    </div>

    <div class="card">
      <div class="card-title">Emotion Scores</div>
      <div id="emotionBars"><div style="color:#4a5568;font-size:12px;text-align:center;padding:10px;">Start detection to see scores</div></div>
    </div>

    <div class="card">
      <div class="card-title">Session Stats</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
        <div style="text-align:center;padding:10px;border-radius:8px;background:rgba(252,73,73,0.08);">
          <div style="font-size:20px;font-weight:700;color:#fc8181;" id="sHigh">0</div>
          <div style="font-size:10px;color:#fc8181;">HIGH</div>
        </div>
        <div style="text-align:center;padding:10px;border-radius:8px;background:rgba(246,173,85,0.08);">
          <div style="font-size:20px;font-weight:700;color:#f6ad55;" id="sMed">0</div>
          <div style="font-size:10px;color:#f6ad55;">MEDIUM</div>
        </div>
        <div style="text-align:center;padding:10px;border-radius:8px;background:rgba(104,211,145,0.08);">
          <div style="font-size:20px;font-weight:700;color:#68d391;" id="sLow">0</div>
          <div style="font-size:10px;color:#68d391;">LOW</div>
        </div>
        <div style="text-align:center;padding:10px;border-radius:8px;background:rgba(102,126,234,0.08);">
          <div style="font-size:20px;font-weight:700;color:#667eea;" id="sTotal">0</div>
          <div style="font-size:10px;color:#667eea;">TOTAL</div>
        </div>
      </div>
    </div>
  </div>
</div>

<canvas id="captureCanvas" style="display:none;"></canvas>

<script>
var video=document.getElementById(\'video\');
var overlay=document.getElementById(\'overlay\');
var cap=document.getElementById(\'captureCanvas\');
var ctx=overlay.getContext(\'2d\');
var capCtx=cap.getContext(\'2d\');
var stream=null,detecting=false,timer=null;
var frameCount=0,lastFps=Date.now();
var sH=0,sM=0,sL=0,sT=0;
var EMOTIONS=[\'angry\',\'disgusted\',\'fearful\',\'happy\',\'neutral\',\'sad\',\'surprised\'];
var EC={angry:\'#fc8181\',disgusted:\'#b794f4\',fearful:\'#f6ad55\',happy:\'#68d391\',neutral:\'#63b3ed\',sad:\'#90cdf4\',surprised:\'#fbd38d\'};

function getCookie(n){var m=document.cookie.match(\'(^|;) ?\'+n+\'=([^;]*)(;|$)\');return m?m[2]:null;}

async function startDetection(){
  try{
    stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480},audio:false});
    video.srcObject=stream;
    document.getElementById(\'noCamera\').style.display=\'none\';
    document.getElementById(\'startBtn\').style.display=\'none\';
    document.getElementById(\'stopBtn\').style.display=\'block\';
    document.getElementById(\'statusText\').textContent=\'Status: Running\';
    detecting=true;
    timer=setInterval(sendFrame,1500);
  }catch(e){alert(\'Camera error: \'+e.message);}
}

function stopDetection(){
  detecting=false;clearInterval(timer);
  if(stream){stream.getTracks().forEach(function(t){t.stop();});stream=null;}
  video.srcObject=null;
  ctx.clearRect(0,0,overlay.width,overlay.height);
  document.getElementById(\'noCamera\').style.display=\'block\';
  document.getElementById(\'startBtn\').style.display=\'block\';
  document.getElementById(\'stopBtn\').style.display=\'none\';
  document.getElementById(\'statusText\').textContent=\'Status: Stopped\';
}

function sendFrame(){
  if(!detecting||!stream||!video.videoWidth)return;
  cap.width=video.videoWidth;cap.height=video.videoHeight;
  overlay.width=video.videoWidth;overlay.height=video.videoHeight;
  capCtx.drawImage(video,0,0);
  var fd=cap.toDataURL(\'image/jpeg\',0.85);
  fetch(\'/live/frame/\',{
    method:\'POST\',
    headers:{\'Content-Type\':\'application/json\',\'X-CSRFToken\':getCookie(\'csrftoken\')},
    body:JSON.stringify({frame:fd})
  })
  .then(function(r){return r.json();})
  .then(function(data){
    if(!data||data.error)return;
    var results=data.results||[];
    document.getElementById(\'faceText\').textContent=\'Faces: \'+results.length;
    ctx.clearRect(0,0,overlay.width,overlay.height);
    results.forEach(function(r){
      var b=r.bbox;
      var color=r.stress_level===\'High\'?\'rgb(252,129,129)\':r.stress_level===\'Medium\'?\'rgb(246,173,85)\':\'rgb(104,211,145)\';
      ctx.strokeStyle=color;ctx.lineWidth=3;ctx.strokeRect(b[0],b[1],b[2],b[3]);
      var label=r.emotion+\' \'+parseFloat(r.confidence).toFixed(0)+\'%\';
      ctx.font=\'bold 15px Segoe UI\';
      var tw=ctx.measureText(label).width+12;
      ctx.fillStyle=color;ctx.fillRect(b[0],b[1]-24,tw,24);
      ctx.fillStyle=\'#000\';ctx.fillText(label,b[0]+6,b[1]-6);
      updatePanel(r);
    });
    frameCount++;
    var now=Date.now();
    if(now-lastFps>=1000){document.getElementById(\'fpsText\').textContent=\'FPS: \'+frameCount;frameCount=0;lastFps=now;}
  })
  .catch(function(e){console.warn(\'Error:\',e);});
}

function updatePanel(r){
  var color=r.stress_level===\'High\'?\'#fc8181\':r.stress_level===\'Medium\'?\'#f6ad55\':\'#68d391\';
  var conf=parseFloat(r.confidence);
  document.getElementById(\'stressText\').textContent=r.stress_level+\' Stress\';
  document.getElementById(\'stressText\').style.color=color;
  document.getElementById(\'emotionText\').textContent=r.emotion;
  document.getElementById(\'emotionText\').style.color=color;
  document.getElementById(\'confFill\').style.width=conf+\'%\';
  document.getElementById(\'confFill\').style.background=color;
  document.getElementById(\'confText\').textContent=\'Confidence: \'+conf.toFixed(1)+\'%\';
  var scores=r.scores||{};
  var html=\'\';
  EMOTIONS.forEach(function(em){
    var val=parseFloat(scores[em]||0);
    html+=\'<div style="display:flex;align-items:center;margin-bottom:6px;">\'+
      \'<span style="font-size:11px;color:#a0aec0;width:70px;text-transform:capitalize;">\'+em+\'</span>\'+
      \'<div style="flex:1;height:5px;background:rgba(255,255,255,0.06);border-radius:3px;margin:0 8px;">\'+
      \'<div style="height:100%;border-radius:3px;background:\'+EC[em]+\';width:\'+val+\'%;transition:width 0.4s;"></div></div>\'+
      \'<span style="font-size:11px;color:#718096;width:38px;text-align:right;">\'+val.toFixed(1)+\'%</span></div>\';
  });
  document.getElementById(\'emotionBars\').innerHTML=html;
  sT++;
  if(r.stress_level===\'High\')sH++;
  else if(r.stress_level===\'Medium\')sM++;
  else sL++;
  document.getElementById(\'sHigh\').textContent=sH;
  document.getElementById(\'sMed\').textContent=sM;
  document.getElementById(\'sLow\').textContent=sL;
  document.getElementById(\'sTotal\').textContent=sT;
}
</script>
</body>
</html>'''

path = 'stress_app/templates/stress_app/user_live.html'
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Written successfully to:", path)
print("Lines:", content.count('\n'))