# BCREC Voice Agent - Architecture & Status

## Stable Architecture Patterns
1.  **Telephony Bridge**: Use `rtc.AudioStream(sample_rate=8000, num_channels=1)` for built-in resampling to Twilio format.
2.  **Audio Buffering**: Always buffer audio into **20ms packets** (320 bytes of 16-bit PCM) before Mu-Law encoding for Twilio WebSocket stability.
3.  **Agent Logic**: For telephony, bypass the `VoiceAgent` wrapper and use direct `AgentSession` event listeners (`user_input_transcribed`) to manually query the LLM and trigger TTS. This avoids VAD/interruption conflicts in high-latency environments.
4.  **Tuning**: Keep greeting delay low (0.5s) but ensure `allow_interruptions=False` for the initial greeting.

## Current Status (May 31, 2026)
- **Demo Ready**: The agent is production-hardened for a Principal-level demo.
- **Turbo Speed**: Migrated to `llama-3.1-8b-instant` for <400ms brain response.
- **Audio Clarity**: Implemented 20ms chunking + 0.7x gain reduction (Volume) to remove clipping/distortion.
- **Linguistic Precision**: 
    - Forced phonetic spelling (e.g., "A. I. M. L.") for flawless pronunciation.
    - Numbers-to-Words conversion in the target language (English/Hindi/Bengali).
    - Strictly mirrors user language to prevent accidental switching.
- **Cleanup**: Temporary test scripts removed. Stable on branch `v2-telephony-dev`.

## Resumption Checklist
- [ ] Start stable Ngrok tunnel.
- [ ] Update `trigger_call.py` base_url.
- [ ] Launch backend and agent (`direct` mode).
