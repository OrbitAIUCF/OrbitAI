let scene, camera, renderer, controls;
let satellites = [];       // THREE.Mesh spheres
let edgeLines;             // THREE.LineSegments
let frames = [];
let frameIndex = 0;
let playing = true;
const playSpeed = 100;     // ms per frame

init();
loadFrames().then(() => {
  buildStage();
  animate();
  window.addEventListener('keydown', onKey);
});

function init() {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 1, 1e7);
  camera.position.set(0, 0, 20000);

  renderer = new THREE.WebGLRenderer({ antialias:true });
  renderer.setSize(innerWidth, innerHeight);
  document.body.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
}

async function loadFrames() {
  const res = await fetch('exported_frames/all_frames.json');
  frames = await res.json();
}

function buildStage() {
  // one sphere per node
  const sphereGeo = new THREE.SphereGeometry(200, 12, 12);
  const mat = new THREE.MeshBasicMaterial();
  frames[0].nodes.forEach(n => {
    const mesh = new THREE.Mesh(sphereGeo, mat.clone());
    scene.add(mesh);
    satellites.push(mesh);
  });
  // placeholder lines
  const edgeGeo = new THREE.BufferGeometry();
  const maxEdges = frames[0].edges.length;
  const positions = new Float32Array(maxEdges*2*3);
  edgeGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  edgeLines = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial());
  scene.add(edgeLines);
}

function updateStage(frame) {
  // update satellites
  frame.nodes.forEach((n,i) => {
    satellites[i].position.set(n.x, n.y, n.z);
  });

  // update edges
  const posAttr = edgeLines.geometry.getAttribute('position');
  let idx = 0;
  frame.edges.forEach(e => {
    // map attention [0,1] → color green→yellow→red
    const c = new THREE.Color();
    c.setHSL((1 - e.attention) * 0.33, 1, 0.5);
    edgeLines.material.color = c;

    const s = satellites[e.source].position;
    const t = satellites[e.target].position;
    posAttr.array[idx++] = s.x;
    posAttr.array[idx++] = s.y;
    posAttr.array[idx++] = s.z;
    posAttr.array[idx++] = t.x;
    posAttr.array[idx++] = t.y;
    posAttr.array[idx++] = t.z;
  });
  posAttr.needsUpdate = true;

  // update table (first edge as example)
  const info = document.getElementById('infoTable');
  const e = frame.edges[0];
  info.innerHTML = `
    Pair: ${e.source}–${e.target}<br>
    Dist: ${e.distance.toFixed(1)}<br>
    RelVel: ${e.rel_vel.toFixed(1)}<br>
    Risk: ${(e.attention*100).toFixed(0)}%
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
  switch(e.code) {
    case 'Space': playing = !playing; break;
    case 'ArrowRight':
      frameIndex = (frameIndex + 1) % frames.length;
      updateStage(frames[frameIndex]);
      break;
    case 'ArrowLeft':
      frameIndex = (frameIndex - 1 + frames.length) % frames.length;
      updateStage(frames[frameIndex]);
      break;
    case 'KeyF': // faster
      playSpeed = Math.max(10, playSpeed - 20);
      break;
    case 'KeyS': // slower
      playSpeed += 20;
      break;
  }
}
