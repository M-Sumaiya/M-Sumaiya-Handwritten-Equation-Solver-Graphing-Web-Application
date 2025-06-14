let canvas1 = document.getElementById('canvas1');
let canvas2 = document.getElementById('canvas2');
let brushSize = document.getElementById('brushSize');

function setupCanvas(canvas) {
    let ctx = canvas.getContext('2d');
    let painting = false;

    // Set white background initially
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Mouse Events
    canvas.addEventListener('mousedown', (e) => {
        painting = true;
        ctx.beginPath();
        ctx.moveTo(e.offsetX, e.offsetY);
    });

    canvas.addEventListener('mouseup', () => {
        painting = false;
        ctx.closePath();
    });

    canvas.addEventListener('mouseleave', () => {
        painting = false;
        ctx.closePath();
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!painting) return;
        ctx.lineWidth = brushSize.value;
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'black';
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        ctx.beginPath(); // Start a new stroke to prevent connection
        ctx.moveTo(e.offsetX, e.offsetY);
    });
}

// Initialize both canvases
setupCanvas(canvas1);
setupCanvas(canvas2);

// Clear both canvases
function clearCanvases() {
    [canvas1, canvas2].forEach(canvas => {
        let ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    });

    document.getElementById("result").style.display = "none";
    document.getElementById("graph").src = "";
}

// Convert canvas to Blob
function canvasToBlob(canvas) {
    return new Promise(resolve => {
        canvas.toBlob(blob => resolve(blob), 'image/png');
    });
}

// Submit canvases to backend
async function submit() {
    const formData = new FormData();
    formData.append("eq1", await canvasToBlob(canvas1));
    formData.append("eq2", await canvasToBlob(canvas2));

    const res = await fetch("/solve", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    document.getElementById("eq1").textContent = data.equation1;
    document.getElementById("eq2").textContent = data.equation2;
    document.getElementById("solution").textContent = data.solution;
    document.getElementById("graph").src = data.graph_url + "?t=" + new Date().getTime();
    document.getElementById("result").style.display = "block";
}
