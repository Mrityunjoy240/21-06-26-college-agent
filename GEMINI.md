# BCREC Voice Agent - Architecture & Status

## Stable Architecture Patterns
1.  **Telephony Bridge**: Use `rtc.AudioStream(sample_rate=8000, num_channels=1)` for built-in resampling to Twilio format.
2.  **Audio Buffering**: Always buffer audio into **20ms packets** (320 bytes of 16-bit PCM) before Mu-Law encoding for Twilio WebSocket stability.
3.  **Agent Logic**: For telephony, bypass the `VoiceAgent` wrapper and use direct `AgentSession` event listeners (`user_input_transcribed`) to manually query the LLM and trigger TTS. This avoids VAD/interruption conflicts in high-latency environments.
4.  **Tuning**: Keep greeting delay low (0.5s) but ensure `allow_interruptions=False` for the initial greeting.

## Current Status (May 23, 2026)
- **Problem Fixed**: The "silent agent" issue was resolved by fixing bridge frame access and agent timing.
- **Verification**: Logs and raw captures confirm speech data is flowing to Twilio.
- **Implementation**: The logic in `backend/app/api/voice.py` and `scripts/livekit_agent.py` is now production-hardened for telephony.

## Resumption Checklist
- [ ] Start stable Ngrok tunnel.
- [ ] Update `trigger_call.py` base_url.
- [ ] Launch backend and agent (`direct` mode).
