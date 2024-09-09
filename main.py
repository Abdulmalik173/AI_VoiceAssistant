from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import Lib.Ai as aii


app = FastAPI()
ai = aii.AI_Assistant()
conversation_history = []
initial_audio_path = "data/ar.mp3" if aii.arabic == True else "data/en.mp3"
print(initial_audio_path)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.get("/get-audio-first")
async def get_audio_first():
    if not os.path.exists(initial_audio_path):
        raise HTTPException(status_code=404, detail="Initial audio file not found")
    return FileResponse(initial_audio_path, media_type="audio/mpeg")

@app.get("/get-audio")
async def get_audio():
    last_response = conversation_history[-1]["response"]
    file_path = ai.generate_audio(last_response)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/mpeg")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Wait for client to send a message to start transcription
            await websocket.receive_text()
            
            # Start transcription
            transcription = ai.start_transcription()
            await websocket.send_json({"type": "transcription", "content": transcription})
            
            # Generate AI response
            ai_response = ai.generate_ai_response(transcription)
            await websocket.send_json({"type": "response", "content": ai_response})
            
            # Update conversation history
            conversation_history.append({"transcription": transcription, "response": ai_response})
            
            # Notify the client to fetch the new audio file
            await websocket.send_json({"type": "fetch_audio"})
    except WebSocketDisconnect:
        ai.on_close()
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)