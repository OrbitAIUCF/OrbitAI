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

    // 4) Outline cube edges for clarity
    const boxGeo   = new THREE.BoxGeometry(gridSize, gridSize, gridSize);
    const edges    = new THREE.EdgesGeometry(boxGeo);
    const line     = new THREE.LineSegments(
        edges,
        new THREE.LineBasicMaterial({ color: 0x444444, transparent: true, opacity: 0.5 })
    );
    scene.add(line);

    const axesDirections = [
        { dir: new THREE.Vector3(1,0,0), color: 0xff0000,   label: 'X' },
        { dir: new THREE.Vector3(0,1,0), color: 0x00ff00,   label: 'Y' },
        { dir: new THREE.Vector3(0,0,1), color: 0x0000ff,   label: 'Z' },
      ];
      const tickValues = [-7000, 0, 7000];
      axesDirections.forEach(({ dir, color, label }) => {
        const axisGroup = new THREE.Group();
      
        // 1) ArrowHelper in the right color
        const arrow = new THREE.ArrowHelper(
          dir,                           // direction
          new THREE.Vector3(),           // origin
          7000,                          // length
          color,                         // color
          200,                           // headLength
          100                            // headWidth
        );
        axisGroup.add(arrow);
      
        // 2) Axis name at +7500
        const nameSprite = makeTextSprite(label, {
            fontsize: 32,
            backgroundColor: 'rgba(0,0,0,0.5)'
        });
        nameSprite.position.copy(dir.clone().multiplyScalar(7500));
        axisGroup.add(nameSprite);

        // 3) Tick‐mark labels
        const tickSprites = tickValues.map(v => {
            const tickSprite = makeTextSprite(String(v), {
            fontsize: 20,
            backgroundColor: 'rgba(0,0,0,0.3)'
            });
            tickSprite.position.copy(dir.clone().multiplyScalar(v));
            return tickSprite;
        });
        tickSprites.forEach(s => axisGroup.add(s));

        scene.add(axisGroup);
      });

}

function buildStage() {
    const sphereGeo = new THREE.SphereGeometry(200, 12, 12);
    const mat       = new THREE.MeshBasicMaterial();
    frames[0].nodes.forEach(n => {
        const mesh = new THREE.Mesh(sphereGeo, mat.clone());
        scene.add(mesh);

        // create a label with the sat id
        const div = document.createElement('div');
        div.className = 'label';
        div.textContent = n.id;    // e.g. the sat_id or index
        
        div.style.marginTop = '-1em';
        const label = new CSS2DObject(div);
        mesh.add(label);

        satellites.push(mesh);
    });

    
    const maxEdges = frames[0].edges.length;
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
    // 1) update the HTML overlays
    document.getElementById('timeLabel').textContent = formatTime(frame.timestamp);
    updateTable(frame.edges);

    // 2) then move satellites & edges…
    frame.nodes.forEach((n,i) => {
        satellites[i].position.set(n.x, n.y, n.z);
    });

    // Update and color edges
    const posAttr   = edgeLines.geometry.getAttribute('position');
    const colorAttr = edgeLines.geometry.getAttribute('color');

    let idx = 0;
    let cidx = 0;

    frame.edges.forEach(e => {
        const s = satellites[e.source].position;
        const t = satellites[e.target].position;

        // Position
        posAttr.array[idx++] = s.x; posAttr.array[idx++] = s.y; posAttr.array[idx++] = s.z;
        posAttr.array[idx++] = t.x; posAttr.array[idx++] = t.y; posAttr.array[idx++] = t.z;

        // Color (same color for both vertices of this line)
        const c = new THREE.Color();
        c.setHSL((1 - e.attention) * 0.33, 1, 0.5);
        colorAttr.array[cidx++] = c.r;
        colorAttr.array[cidx++] = c.g;
        colorAttr.array[cidx++] = c.b;
        colorAttr.array[cidx++] = c.r;
        colorAttr.array[cidx++] = c.g;
        colorAttr.array[cidx++] = c.b;
    });

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

}

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
