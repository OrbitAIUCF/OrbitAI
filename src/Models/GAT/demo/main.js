import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.158.0/examples/jsm/controls/OrbitControls.js';


let scene, camera, renderer, controls;
let satellites = [];
let edgeLines;
let frames = [], frameIndex = 0, playing = true, playSpeed = 100;
const infoEl = document.getElementById('infoTable');

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
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 1, 1e7);
  camera.position.set(0, 0, 20000);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(innerWidth, innerHeight);
  document.body.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
}

function buildStage() {
  const sphereGeo = new THREE.SphereGeometry(200, 12, 12);
  const mat       = new THREE.MeshBasicMaterial();
  frames[0].nodes.forEach(n => {
    const mesh = new THREE.Mesh(sphereGeo, mat.clone());
    scene.add(mesh);
    satellites.push(mesh);
  });

  const maxEdges = frames[0].edges.length;
  const edgeGeo  = new THREE.BufferGeometry();
  const positions = new Float32Array(maxEdges * 2 * 3);
  edgeGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  edgeLines = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial());
  scene.add(edgeLines);
}

function updateStage(frame) {
  // Move satellites
  frame.nodes.forEach((n,i) => {
    satellites[i].position.set(n.x, n.y, n.z);
  });

  // Update and color edges
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

  // Update info panel (first edge as example)
  const e = frame.edges[0];
  infoEl.innerHTML = `
    Pair: ${e.source}–${e.target}<br>
    Dist: ${e.distance.toFixed(1)}<br>
    RelVel: ${e.rel_vel.toFixed(1)}<br>
    Risk: ${(e.attention * 100).toFixed(0)}%
  `;
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);

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
