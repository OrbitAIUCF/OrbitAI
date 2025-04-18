import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.158.0/examples/jsm/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'https://cdn.jsdelivr.net/npm/three@0.158.0/examples/jsm/renderers/CSS2DRenderer.js';


let scene, camera, renderer, controls, labelRenderer;
let satellites = [];
let edgeLines;
let frames = [], frameIndex = 0, playing = true, playSpeed = 100;
const infoEl = document.getElementById('timeLabel');
const raycaster = new THREE.Raycaster();
const mouse     = new THREE.Vector2();

window.addEventListener('click', onClick, false);

function onClick(event) {
  // calculate pointer coords [-1,1]
  mouse.x = ( event.clientX / window.innerWidth ) *  2 - 1;
  mouse.y = ( event.clientY / window.innerHeight ) * -2 + 1;
  raycaster.setFromCamera(mouse, camera);

  // test against your satellite meshes
  const intersects = raycaster.intersectObjects(satellites, true);
  if (intersects.length) {
    const mesh = intersects[0].object;
    const idx  = satellites.indexOf(mesh);
    const info = frames[frameIndex].nodes[idx];
    alert(`Sat ${info.id}\nPos: ${info.x.toFixed(1)}, ${info.y.toFixed(1)}, ${info.z.toFixed(1)}`);
  }
}

function makeTextSprite(text, { fontsize=24, fontface='Arial', borderColor='black', backgroundColor='rgba(0,0,0,0)' }={}) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    ctx.font = `${fontsize}px ${fontface}`;
    const metrics = ctx.measureText(text);
    canvas.width  = metrics.width;
    canvas.height = fontsize * 1.2;

    // background
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // text
    ctx.fillStyle = 'white';
    ctx.fillText(text, 0, fontsize);
    const texture = new THREE.CanvasTexture(canvas);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false }));
    sprite.scale.set(canvas.width, canvas.height, 1);
    return sprite;
}
  
init();
loadFrames();        // <-- calls the single loader below
window.addEventListener('keydown', onKey);

async function loadFrames() {
    try {
        const res = await fetch('./exported_frames/all_frames.json');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        frames = await res.json();
        infoEl.innerHTML = `✅ Loaded ${frames.length} frames`;
        buildStage();
        updateStage(frames[0]);
        animate();
    } catch (err) {
        console.error('Failed to load frames:', err);
        infoEl.innerHTML = `❌ Error loading frames: ${err.message}`;
    }
}

// AXIS TICKS 
function addAxisTicks() {
    const step       = 2000;     // tick spacing  (km → world units)
  const minVal     = -7000;
  const maxVal     =  7000;
  const tickSize   = 200;      // length of the little tick mark (world units)
  const labelGap   = 300;      // distance the number sits away from the axis

  /* build one thin line segment */
  function makeTick(p1, p2, color) {
    const g = new THREE.BufferGeometry().setFromPoints([p1, p2]);
    const m = new THREE.LineBasicMaterial({ color });
    scene.add(new THREE.Line(g, m));
  }

  /* build the numeric text label */
  function makeTickLabel(text, pos) {
    const el       = document.createElement('div');
    el.className   = 'tickLabel';
    el.textContent = text;
    const label    = new CSS2DObject(el);
    label.position.copy(pos);
    scene.add(label);
  }

  /* ---------- X‑axis ---------- */
  for (let v = minVal; v <= maxVal; v += step) {
    const base  = new THREE.Vector3(v, -7000, -7000);
    const up    = new THREE.Vector3(0, 1, 0).setLength(tickSize / 2);
    makeTick(base.clone().sub(up), base.clone().add(up), 0xff0000);
    makeTickLabel(v, base.clone().sub(up).sub(new THREE.Vector3(0, labelGap, 0)));
  }
  // extra 0‑km tick if it wasn’t reached by the loop
  if ((0 - minVal) % step !== 0) {
    const base0 = new THREE.Vector3(0, -7000, -7000);
    const up0   = new THREE.Vector3(0, 1, 0).setLength(tickSize / 2);
    makeTick(base0.clone().sub(up0), base0.clone().add(up0), 0xff0000);
    makeTickLabel(0, base0.clone().sub(up0).sub(new THREE.Vector3(0, labelGap, 0)));
  }

  /* ---------- Y‑axis ---------- */
  for (let v = minVal + step; v <= maxVal; v += step) {   // <‑ skip −7000
    const base  = new THREE.Vector3(-7000, v, -7000);
    const right = new THREE.Vector3(1, 0, 0).setLength(tickSize / 2);
    makeTick(base.clone().sub(right), base.clone().add(right), 0x00ff00);
    makeTickLabel(v, base.clone().sub(right).sub(new THREE.Vector3(labelGap, 0, 0)));
  }
  // explicit 0‑km tick
  const baseY0 = new THREE.Vector3(-7000, 0, -7000);
  const right0 = new THREE.Vector3(1, 0, 0).setLength(tickSize / 2);
  makeTick(baseY0.clone().sub(right0), baseY0.clone().add(right0), 0x00ff00);
  makeTickLabel(0, baseY0.clone().sub(right0).sub(new THREE.Vector3(labelGap, 0, 0)));

  /* ---------- Z‑axis ---------- */
  for (let v = minVal; v <= maxVal; v += step) {
    const base  = new THREE.Vector3(-7000, -7000, v);
    const up    = new THREE.Vector3(0, 1, 0).setLength(tickSize / 2);
    makeTick(base.clone().sub(up), base.clone().add(up), 0x0000ff);
    makeTickLabel(v, base.clone().sub(up).sub(new THREE.Vector3(0, labelGap, 0)));
  }
  // extra 0‑km tick for Z
  if ((0 - minVal) % step !== 0) {
    const baseZ0 = new THREE.Vector3(-7000, -7000, 0);
    const upZ0   = new THREE.Vector3(0, 1, 0).setLength(tickSize / 2);
    makeTick(baseZ0.clone().sub(upZ0), baseZ0.clone().add(upZ0), 0x0000ff);
    makeTickLabel(0, baseZ0.clone().sub(upZ0).sub(new THREE.Vector3(0, labelGap, 0)));
  }
}
  
function init() {
    // Scene & background
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101020);    // dark navy, not pure black

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(10000, 10000, 10000);
    scene.add(dirLight);

    // Camera
    camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 1, 1e7);
    camera.position.set(0, 0, 20000);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(innerWidth, innerHeight);
    document.body.appendChild(renderer.domElement);

    // CSS2D labels
    labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(innerWidth, innerHeight);
    labelRenderer.domElement.style.position = 'absolute';
    labelRenderer.domElement.style.top      = '0';
    labelRenderer.domElement.style.pointerEvents = 'none';
    document.body.appendChild(labelRenderer.domElement);

    // Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // 7000km grid
    const gridSize     = 14000;   // spans -7000 → +7000
    const gridDivs     = 7;
    const gridColor1   = 0x444444;
    const gridColor2   = 0x222222;

    // 1) XZ plane (ground plane)
    const gridXZ = new THREE.GridHelper(gridSize, gridDivs, gridColor1, gridColor2);
    scene.add(gridXZ);

    // 2) YZ plane (rotate XZ around Z to stand it up on YZ)
    const gridYZ = gridXZ.clone();
    gridYZ.rotation.z = Math.PI / 2;
    scene.add(gridYZ);

    // 3) XY plane (rotate XZ around X to stand it up on XY)
    const gridXY = gridXZ.clone();
    gridXY.rotation.x = Math.PI / 2;
    scene.add(gridXY);

    gridXZ.position.y = -7000;
    gridYZ.position.x = -7000;
    gridXY.position.z = -7000;

    // 4) Outline cube edges for clarity
    // const boxGeo   = new THREE.BoxGeometry(gridSize, gridSize, gridSize);
    // const edges    = new THREE.EdgesGeometry(boxGeo);
    // const line     = new THREE.LineSegments(
    //     edges,
    //     new THREE.LineBasicMaterial({ color: 0x444444, transparent: true, opacity: 0.5 })
    // );
    // scene.add(line);

    const axes = [
        {
          // X‐axis
          arrowDir:  new THREE.Vector3(1,0,0),
          labelPos:  new THREE.Vector3( 7200,  -7200,  -7200),
          color:     0xff0000,
          name:      'X (km)',
        },
        {
          // Y‐axis
          arrowDir:  new THREE.Vector3(0,1,0),
          labelPos:  new THREE.Vector3( -7200,  7200,  -7200),
          color:     0x00ff00,
          name:      'Y (km)',
        },
        {
          // Z‐axis
          arrowDir:  new THREE.Vector3(0,0,1),
          labelPos:  new THREE.Vector3( -7200,  -7200,  7500),
          color:     0x0000ff,
          name:      'Z (km)',
        },
      ];
      
      const OFFSET = 1350   // push label 750 wu beyond the arrow tip
      axes.forEach(axis=>{
        // ——— ArrowHelper ———
        const origin = new THREE.Vector3(-7000, -7000, -7000);
        const arrow  = new THREE.ArrowHelper(
          axis.arrowDir,  // direction
          origin,         // arrow base
          14000,          // length
          axis.color,     // color
          200,            // headLength
          100             // headWidth
        );
        scene.add(arrow);
      
        addAxisTicks();
        // ——— CSS2D label ———
        const div = document.createElement('div');
        div.className   = 'axisLabel';
        div.textContent = axis.name;
        // style it purely in CSS:
        div.style.color     = '#ffffff';
        div.style.fontSize  = '24px';
        div.style.fontWeight= 'bold';
        div.style.pointerEvents = 'none';
        
        const endPoint = origin.clone().add(axis.arrowDir.clone().multiplyScalar(14000 + OFFSET)); //arrow tip
        const label    = new CSS2DObject(div);
        label.position.copy(endPoint).addScaledVector(axis.arrowDir, 250);           // 250 wu past tip
        scene.add(label);
       });      
}

function buildStage() {
    const sphereGeo = new THREE.SphereGeometry(200, 12, 12);
    const mat       = new THREE.MeshBasicMaterial();
    const SPHERE_R   = 200;          // radius used in your SphereGeometry
    const LABEL_GAP  = 200;           // world‑unit gap above the sphere

    frames[0].nodes.forEach(n => {
        const mesh = new THREE.Mesh(sphereGeo, mat.clone());
        scene.add(mesh);

        // create a label with the sat id
        const div = document.createElement('div');
        div.className = 'label';
        div.textContent = n.id;    // e.g. the sat_id or index
        
        //div.style.marginTop = '-1em';
        const label2d  = new CSS2DObject(div);
        label2d.position.set(0, SPHERE_R + LABEL_GAP, 0);
        mesh.add(label2d);

        satellites.push(mesh);
    });

    //const maxEdges = frames[0].edges.length;
    const maxEdges = Math.max(...frames.map(f => f.edges.length));

    const edgeGeo  = new THREE.BufferGeometry();
    const positions = new Float32Array(maxEdges * 2 * 3);
    edgeGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const colors = new Float32Array(maxEdges * 2 * 3);  // 2 vertices per edge, 3 floats per color (r,g,b)
    edgeGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    //edgeLines = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial());
    edgeLines = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ vertexColors: true }));

    scene.add(edgeLines);
}

function updateStage(frame) {
    // grab the buffers for this frame
    const posAttr   = edgeLines.geometry.getAttribute('position');
    const colorAttr = edgeLines.geometry.getAttribute('color');
  
    // 1) update your HTML overlays
    document.getElementById('timeLabel').textContent = formatTime(frame.timestamp);
    updateTable(frame.edges);
  
    // 2) move satellites…
    frame.nodes.forEach((n,i) => {
      satellites[i].position.set(n.x, n.y, n.z);
    });
  
    // 3) now you can safely iterate frame.edges and write into posAttr.array & colorAttr.array
    let idx = 0, cidx = 0;
    frame.edges.forEach(e => {
      const s = satellites[e.source].position;
      const t = satellites[e.target].position;
  
      // positions
      posAttr.array[idx++] = s.x;
      posAttr.array[idx++] = s.y;
      posAttr.array[idx++] = s.z;
      posAttr.array[idx++] = t.x;
      posAttr.array[idx++] = t.y;
      posAttr.array[idx++] = t.z;
  
      // discrete bright color
      let c;
      if      (e.attention > 0.66) c = new THREE.Color(0xff0000);  // red
      else if (e.attention > 0.33) c = new THREE.Color(0xffff00);  // yellow
      else                          c = new THREE.Color(0x00ff00);  // green
  
      // both verts same color
      colorAttr.array[cidx++] = c.r;
      colorAttr.array[cidx++] = c.g;
      colorAttr.array[cidx++] = c.b;
      colorAttr.array[cidx++] = c.r;
      colorAttr.array[cidx++] = c.g;
      colorAttr.array[cidx++] = c.b;
    });
  
    // zero out any unused slots
    for (; idx   < posAttr.array.length;   idx++)   posAttr.array[idx]   = 0;
    for (; cidx  < colorAttr.array.length; cidx++)  colorAttr.array[cidx] = 0;
  
    posAttr.needsUpdate   = true;
    colorAttr.needsUpdate = true;
  }
  
    // Update and color edges
    /*
    const posAttr = edgeLines.geometry.getAttribute('position');
    let idx = 0;
    frame.edges.forEach(e => {
        // green→yellow→red based on attention
        const c = new THREE.Color();
        c.setHSL((1 - e.attention) * 0.33, 1, 0.5);
        edgeLines.material.color = c;

        const s = satellites[e.source].position;
        const t = satellites[e.target].position;
        posAttr.array[idx++] = s.x; posAttr.array[idx++] = s.y; posAttr.array[idx++] = s.z;
        posAttr.array[idx++] = t.x; posAttr.array[idx++] = t.y; posAttr.array[idx++] = t.z;
    });
    posAttr.needsUpdate = true;
    */

function formatTime(ts){
    const dt = new Date(ts);
    return new Intl.DateTimeFormat('en-US',{
        month:'long', day:'numeric', year:'numeric',
        hour:'numeric', minute:'2-digit', second:'2-digit',
        hour12:false
    }).format(dt);
}

function riskColor(a){
    return a>0.66?'red':a>0.33?'yellow':'green';
}
function riskLabel(a){
    return a>0.66?'HIGH': a>0.33?'MEDIUM':'LOW';
}

function updateTable(edges) {
    // 1) dedupe by min(source,target)
    const seen = new Set();
    edges = edges.filter(e => {
      const key = [e.source,e.target].sort().join('-');
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    // 2) sort descending by attention
    edges.sort((a,b) => b.attention - a.attention);
    // 3) build rows
    const tbody = document.querySelector('#riskTable tbody');
    tbody.innerHTML = '';
    console.log("New set of Edges")
    edges.forEach(e => {
      console.log(e)
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${Number(e.source)+1}</td>
        <td>${e.distance.toFixed(1)}</td>
        <td>${e.rel_vel.toFixed(1)}</td>
        <td style="color:${riskColor(e.attention)}">
          ${riskLabel(e.attention)}
        </td>
        <td>${Number(e.target)+1}</td>
      `;
      tbody.appendChild(tr);
    });
  }
  

  
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);

    if (playing && frames.length) {
        const now = performance.now();
        if (!animate.last || now - animate.last > playSpeed) {
        frameIndex = (frameIndex + 1) % frames.length;
        updateStage(frames[frameIndex]);
        animate.last = now;
        }
    }
}

function onKey(e) {
    switch (e.code) {
        case 'Space':
            playing = !playing;
            break;
        case 'ArrowRight':
            frameIndex = (frameIndex + 1) % frames.length;
            updateStage(frames[frameIndex]);
            break;
        case 'ArrowLeft':
            frameIndex = (frameIndex - 1 + frames.length) % frames.length;
            updateStage(frames[frameIndex]);
            break;
        case 'KeyF':
            playSpeed = Math.max(10, playSpeed - 20);
            break;
        case 'KeyS':
            playSpeed += 20;
            break;
    }
}

// Button event listeners
document.getElementById('playBtn').onclick  = () => playing = true;
document.getElementById('pauseBtn').onclick = () => playing = false;
document.getElementById('fasterBtn').onclick = () => playSpeed = Math.max(10, playSpeed - 20);
document.getElementById('slowerBtn').onclick = () => playSpeed += 20;
document.getElementById('resetCam').onclick = () => controls.reset();

window.addEventListener('keydown', e=>{
    if (e.key==='p') playing = !playing;
    if (e.key==='r') controls.reset();
  });
