# 🎯 BCREC Principal Demo: Master Control Board

## 📢 THE MISSION
Demonstrate a production-hardened AI Receptionist that is seamless on Phone (Telephony) and Web (Rich Chat).

---

## 🏗️ SYSTEM STATUS (The Agents)

### 🤖 Agent 1: Telephony Edge
- **Status:** 🟢 STABLE (MIGRATED)
- **Goal:** Moved from `AgentSession` to `voice.Agent`.
- **Features:** 20ms packet chunking, 0.7x gain, phonetic lexicon applied.

### 🎨 Agent 2: Visual Anchor (Web)
- **Status:** 🟢 STABLE (POLISHED)
- **Goal:** Added Markdown rendering and word-by-word streaming.
- **Features:** Tables for fees, bold headers, real-time typing feel.

### 🧠 Agent 3: Knowledge Guard (RAG)
- **Status:** 🟢 STABLE
- **Goal:** Implement Hybrid Search (BM25) to prevent hallucinations.
- **Requirements:** Must distinguish between similar names/departments.

---

## 📖 LEXICON (Permanent Pronunciation)
- **BCREC**: B. C. R. E. C.
- **MAKAUT**: Ma-Kaut
- **AIML**: A. I. M. L.
- **CSE**: C. S. E.
- **Jemua**: Jay-moo-ah

---

## 🚀 DEMO CHECKLIST
- [ ] LiveKit v1.5.12 Migration (voice.Agent)
- [ ] Markdown Tables in Frontend
- [ ] 20ms Audio Buffering Stable
- [ ] Hybrid Search (BM25 + Vector) Active
- [ ] Barge-in (Interruptions) Tested
