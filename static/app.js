const canvas = document.getElementById('visualizer');
const ctx = canvas.getContext('2d');
const playButton = document.getElementById('playButton');
const audioInfoList = document.getElementById('audioInfoList');

let audioContext;
let analyser;
let source;
let animationId;
let isPlaying = false;
let ws= new WebSocket('ws://localhost:8000/ws');

playButton.addEventListener('click', async function() {
    if (isPlaying) {
        stopAudio();
    } else {
        await playAudio();
    }
});

async function playAudio() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 1024;
    }

    try {
        await fetchAndPlayAudio('/get-audio-first');

        ws.onmessage = async function(event) {
            const data = JSON.parse(event.data);
            console.log('WebSocket message:', data);
            
            if (data.type === "fetch_audio") {
                await fetchAndPlayAudio('/get-audio');
            } else {
                const listItem = document.createElement('li');
                listItem.textContent = data.content;
                audioInfoList.prepend(listItem);
                if (audioInfoList.children.length > 5) {
                    audioInfoList.removeChild(audioInfoList.lastChild);
                }
            }
        };
        ws.onclose = function(event) {
            console.log('WebSocket closed:', event);
            stopAudio();
        };
        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
            stopAudio();
        };

        isPlaying = true;
        playButton.textContent = 'Stop Audio';
        playButton.classList.remove('bg-blue-500', 'hover:bg-blue-600');
        playButton.classList.add('bg-red-500', 'hover:bg-red-600');
    } catch (error) {
        console.error('Error fetching or playing audio:', error);
    }
}

async function fetchAndPlayAudio(url) {
    let response = await fetch(url);
    let arrayBuffer = await response.arrayBuffer();
    let audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    if (source) {
        source.disconnect();
        console.log('Disconnected source:', source);
    }
    source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(analyser);
    analyser.connect(audioContext.destination);
    source.start(0);
    visualize();
    source.onended = function() {
        ws.send('start_transcription');
    };
}

function stopAudio() {
    if (source) {
        source.stop();
    }
    if (animationId) {
        cancelAnimationFrame(animationId);
    }
    isPlaying = false;
    playButton.textContent = 'Play Audio';
    playButton.classList.remove('bg-red-500', 'hover:bg-red-600');
    playButton.classList.add('bg-blue-500', 'hover:bg-blue-600');

    if (ws) {
        ws.close();
    }
}

function visualize() {
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    let squares = [
        { targetHeight: 0, currentHeight: 0, color: '#E1E3FF' },
        { targetHeight: 0, currentHeight: 0, color: '#D3CCFF' },
        { targetHeight: 0, currentHeight: 0, color: '#C1B9FF' },
        { targetHeight: 0, currentHeight: 0, color: '#B4A7FF' }
    ];

    function draw() {
        animationId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = 'rgba(31, 41, 55, 0.2)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const squareWidth = canvas.width / 5;
        const maxHeight = canvas.height - 40;

        squares.forEach((square, index) => {
            const startFreq = index * (bufferLength / 16);
            const endFreq = (index + 1) * (bufferLength / 16);
            const avgFrequency = dataArray.slice(startFreq, endFreq).reduce((a, b) => a + b) / (bufferLength / 16);
            
            square.targetHeight = (avgFrequency / 255) * maxHeight;
            square.currentHeight += (square.targetHeight - square.currentHeight) * 0.15;

            const x = index * squareWidth + squareWidth / 2;
            const y = canvas.height - square.currentHeight - 20;

            ctx.beginPath();
            ctx.moveTo(x + 10, y);
            ctx.lineTo(x + squareWidth - 20, y);
            ctx.quadraticCurveTo(x + squareWidth - 10, y, x + squareWidth - 10, y + 10);
            ctx.lineTo(x + squareWidth - 10, canvas.height - 30);
            ctx.quadraticCurveTo(x + squareWidth - 10, canvas.height - 20, x + squareWidth - 20, canvas.height - 20);
            ctx.lineTo(x + 10, canvas.height - 20);
            ctx.quadraticCurveTo(x, canvas.height - 20, x, canvas.height - 30);
            ctx.lineTo(x, y + 10);
            ctx.quadraticCurveTo(x, y, x + 10, y);
            ctx.closePath();

            const gradient = ctx.createLinearGradient(x, y, x, canvas.height - 20);
            gradient.addColorStop(0, square.color);
            gradient.addColorStop(1, 'rgba(255, 255, 255, 0.1)');
            ctx.fillStyle = gradient;
            ctx.fill();

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.shadowColor = square.color;
            ctx.shadowBlur = 10;
            ctx.stroke();
            ctx.shadowBlur = 0;
        });
    }

    draw();
}

// Initial WebSocket message to start transcription
ws.onopen = function(event) {
    // ws.send('start_transcription');
};