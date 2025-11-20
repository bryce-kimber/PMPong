# PMPong - (Poor Man's Pong) 
PMPong is a tool that listens through your computer's microphone and wakes you up when your work phone rings. It doesn’t connect to the phone at all. Even if the phone is locked down under strict MDM rules, PMPong still works because it just listens for the ringtone in the room.

### What it does
- Records a short sample of your ringtone
- Builds an audio fingerprint
- Listens in one-second chunks for that same pattern
- Sounds a loud air-raid style alarm when it hears it

### How to run it
`pip install -r requirements.txt`  
`python app.py`

Then open your browser and go to:
`http://127.0.0.1:5000`

### How to use it
1. Play your ringtone near the computer's mic and press “Record Ringtone.”
2. Once the fingerprint is saved, press “Start Detection.”
3. When your phone rings, PMPong will sound the alarm.
4. Press “Stop Siren” in the UI if needed.
Everything runs locally and nothing is uploaded or stored beyond the fingerprint file.

### Notes
If PMPong has trouble hearing your ringtone, try re-recording it in a quieter room or placing the phone closer to the microphone.  
On macOS, make sure Python has microphone access. On Linux, you may need PortAudio packages.  
I am a heavy sleeper so I created this to make sure I don't sleep through my phone ringing.
