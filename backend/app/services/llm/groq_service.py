"""
Groq LLM Service for BCREC Voice Agent Prototype

This module provides a pure LLM-based query service using Groq API.
It injects the full knowledge base into the prompt and uses the LLM to handle
all query variations naturally.
"""
import json
import logging
import re
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq not installed. Groq service will be unavailable.")

from app.config import settings


class GroqService:
    """
    Pure LLM service using Groq API.
    
    This service:
    1. Loads the full knowledge base
    2. Formats it into the system prompt
    3. Passes conversation history
    4. Uses Groq (Llama/Mixtral) to generate accurate responses
    """
    
    SYSTEM_PROMPT = """You are Ritu (if Bengali) or Shubh (if English/Hindi), a warm admission counselor at Dr. B.C. Roy Engineering College.

STRICT RULES:
1. GREETING RULE: If user just says "Hi/Hello", respond with a friendly 1-sentence greeting.
2. NO REPETITION: Do NOT introduce yourself if already done.
3. LANGUAGE & SCRIPT: 
   - User speaks Bengali -> Respond in Bengali Script (বাংলা). Use Bengali digits (০, ১, ২...).
   - User speaks Hindi -> Respond in Hindi Script (हिन्दी). Use standard digits (0, 1, 2...).
   - User speaks English -> Respond in English. Use standard digits (0, 1, 2...).
4. CRITICAL BREVITY: MAXIMUM 2 SENTENCES. NO EXCEPTIONS. If you write more, the system will cut you off.
5. PHONETICS: Acronyms must have spaces (C S E, A I M L). In Bengali: CSE="সি এস ই", IT="আই টি".
6. CURRENCY: English/Hindi: "Rs. 80,500". Bengali: "৮০,৫০০ টাকা" (NEVER use "রুপি").
7. STRICT ACCURACY: Do NOT mix up departments. If asked about AIML faculty, only state AIML faculty. Never combine faculties from different branches (like DS).
8. FACULTY FALLBACK: If asked about general faculty members, explicitly say: "I currently only have the HOD's name. For the rest, please contact the college at 0343-2501353."
9. ACRONYMS IN HINDI/BENGALI: ALWAYS write acronyms (WBJEE, CSE, AIML, BTECH) in English letters. Do NOT transliterate them into Hindi or Bengali script (e.g. NEVER write वीजेईई or সিএসই).
10. DIRECTIONS FALLBACK: If asked how to travel or reach the college, DO NOT invent bus numbers or routes. Only state the exact address (Jemua Road, Fuljhore, Durgapur).

MAX 2 SENTENCES. BE SNAPPY."""

    def __init__(self):
        # Switching to 8b for ultra-low latency and higher rate limits for demo
        self.model = "llama-3.1-8b-instant" 
        self.temperature = 0.1 
        self.max_tokens = 1024 # Increased to 1024 because Bengali characters use many tokens
        self.knowledge_base = self._load_knowledge_base()
        
        # Directly use the client from settings
        self.client = settings.groq_client
        
        if self.client:
            logger.info("GroqService successfully connected to settings client")
        else:
            logger.warning(f"GroqService: No client found in settings. API Key: {'present' if settings.groq_api_key else 'MISSING'}")
    
    def _load_knowledge_base(self) -> str:
        """Load and format the knowledge base for the prompt"""
        try:
            import os
            kb_path = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "..", "data", "knowledge_base", "combined_kb.json"
            )
            kb_path = os.path.normpath(kb_path)
            
            with open(kb_path, 'r', encoding='utf-8') as f:
                kb_data = json.load(f)
            
            formatted_kb = self._format_knowledge_base(kb_data)
            return formatted_kb
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            return ""
    
    def _format_knowledge_base(self, kb_data: Dict[str, Any]) -> str:
        """Format knowledge base into readable text for the prompt"""
        lines = []
        
        # College Info
        if "college" in kb_data:
            college = kb_data["college"]
            lines.append("=== COLLEGE INFORMATION ===")
            lines.append(f"Name: {college.get('name', 'Dr. B.C. Roy Engineering College')}")
            lines.append(f"Location: {college.get('location', 'Durgapur, West Bengal')}")
            lines.append(f"Established: {college.get('established', '2000')}")
            if college.get('naac'):
                lines.append(f"NAAC Grade: {college.get('naac', {}).get('grade', 'B+')}")
            if college.get('contact'):
                contact = college.get('contact', {})
                lines.append(f"Phone: {contact.get('phones', ['0343-2501353'])[0]}")
                lines.append(f"Email: {contact.get('email', 'info@bcrec.ac.in')}")
                lines.append(f"Website: {contact.get('website', 'www.bcrec.ac.in')}")
            lines.append("")
        
        # Courses
        if "courses" in kb_data:
            # B.Tech Courses
            if "btech" in kb_data["courses"]:
                lines.append("=== B.TECH COURSES ===")
                for code, info in kb_data["courses"]["btech"].items():
                    lines.append(f"\n{code} - {info.get('full_name', code)}:")
                    lines.append(f"  Intake: {info.get('intake', 'N/A')} students")
                    fees = info.get('fees', {})
                    if fees:
                        lines.append(f"  Total Fee: Rs. {fees.get('total', 'N/A')}")
                    cutoff = info.get('cutoff', {})
                    if cutoff:
                        lines.append(f"  WBJEE 2025 Cutoff: {cutoff.get('2025', 'N/A')}")
                    placement = info.get('placement', {})
                    if placement:
                        lines.append(f"  Placement: {placement.get('2024-25', 'N/A')} ({placement.get('avg_lpa', '')} LPA avg)")
                lines.append("")
            
            # MBA
            if "mba" in kb_data["courses"]:
                mba = kb_data["courses"]["mba"]
                lines.append("=== MBA COURSE ===")
                lines.append(f"Intake: {mba.get('intake', 'N/A')} students")
                lines.append(f"Duration: {mba.get('duration', '2 years')}")
                lines.append(f"Total Fee: Rs. {mba.get('fees', {}).get('total', 'N/A')}")
                if mba.get('scholarship'):
                    lines.append(f"Scholarship: {mba['scholarship']}")
                lines.append("")
            
            # MCA
            if "mca" in kb_data["courses"]:
                mca = kb_data["courses"]["mca"]
                lines.append("=== MCA COURSE ===")
                lines.append(f"Intake: {mca.get('intake', 'N/A')} students")
                lines.append(f"Duration: {mca.get('duration', '2 years')}")
                lines.append(f"Total Fee: Rs. {mca.get('fees', {}).get('total', 'N/A')}")
                lines.append(f"Eligibility: {mca.get('eligibility', 'JECA qualified')}")
                lines.append("")
            
            # M.Tech
            if "mtech" in kb_data["courses"]:
                mtech = kb_data["courses"]["mtech"]
                lines.append("=== M.TECH COURSES ===")
                lines.append(f"Programs: {', '.join(mtech.get('programs', []))}")
                lines.append(f"Intake: {mtech.get('intake', 'N/A')} students")
                lines.append(f"Duration: {mtech.get('duration', '2 years')}")
                lines.append(f"Total Fee: Rs. {mtech.get('fees', {}).get('total', 'N/A')}")
                if mtech.get('stipend'):
                    lines.append(f"Stipend: {mtech['stipend']}")
                lines.append("")
        
        # College Accreditations
        if "college" in kb_data:
            college = kb_data["college"]
            if college.get('naac'):
                naac = college['naac']
                lines.append("=== COLLEGE ACCREDITATIONS ===")
                lines.append(f"NAAC Grade: {naac.get('grade', 'B+')} (CGPA: {naac.get('cgpa', '2.83')})")
            if college.get('nba_programs'):
                lines.append(f"NBA Accredited: {', '.join(college['nba_programs'])}")
            if college.get('autonomous'):
                lines.append("Status: Autonomous Institute (from 2024-25 batch)")
            if college.get('aicte_idea_lab_rank'):
                lines.append(f"AICTE IDEA Lab: {college['aicte_idea_lab_rank']}")
            lines.append("")
        
        # Placements
        if "placements" in kb_data:
            placements = kb_data["placements"]
            lines.append("=== PLACEMENT INFORMATION ===")
            lines.append(f"Overall Placement Rate: {placements.get('overall_rate_2025', '80%+')}")
            lines.append(f"Highest Salary Branch: {placements.get('highest_salary_branch', 'CSE has highest placements')}")
            if placements.get('highest_package'):
                hp = placements['highest_package']
                if isinstance(hp, dict):
                    lines.append(f"Highest Package: Rs. {hp.get('amount', 30)} LPA ({hp.get('company', 'Cyberwissen')})")
                else:
                    lines.append(f"Highest Package: {hp}")
            if placements.get('top_companies_2026'):
                companies = placements['top_companies_2026']
                if isinstance(companies, list) and len(companies) > 0:
                    lines.append("Top Recruiters:")
                    for c in companies[:5]:
                        if isinstance(c, dict):
                            lines.append(f"  - {c.get('company', '')} ({c.get('package', '')})")
            if placements.get('internship'):
                intern = placements['internship']
                if isinstance(intern, dict):
                    lines.append(f"Internship Available: {intern.get('available', 'Yes')}")
                    if intern.get('description'):
                        lines.append(f"  {intern['description']}")
                    if intern.get('software_stipend'):
                        lines.append(f"  Software Stipend: {intern['software_stipend']}")
            if placements.get('training_cell'):
                tc = placements['training_cell']
                if isinstance(tc, dict):
                    lines.append("Placement Training Programs:")
                    if tc.get('programs'):
                        for prog, desc in tc['programs'].items():
                            lines.append(f"  - {prog.title()}: {desc}")
            lines.append("")
        
        # Fees Summary
        if "fees_summary" in kb_data:
            fees = kb_data["fees_summary"]
            lines.append("=== FEE STRUCTURE ===")
            if "semester_wise" in fees:
                sw = fees["semester_wise"]
                lines.append(f"1st Semester: Rs. {sw.get('first', 97525):,}")
                lines.append(f"Semester 2-7: Rs. {sw.get('semesters_2_to_7', '72,425-74,425')}/sem")
                lines.append(f"8th Semester: Rs. {sw.get('eighth', 73425):,}")
                lines.append(f"Total B.Tech: {sw.get('total_description', '~Rs. 6.08 Lakhs')}")
            lines.append("")
        
        # Admission
        if "admission" in kb_data:
            adm = kb_data["admission"]
            lines.append("=== ADMISSION INFORMATION ===")
            if adm.get('eligibility'):
                elig = adm['eligibility']
                if isinstance(elig, dict) and 'btech' in elig:
                    lines.append(f"Eligibility: {elig['btech']}")
            if adm.get('counseling'):
                lines.append(f"Counseling: {adm['counseling']}")
            if adm.get('seat_distribution'):
                lines.append(f"Seat Distribution: {adm['seat_distribution']}")
            lines.append("")
        
        # Scholarships
        if "scholarships" in kb_data:
            sch = kb_data["scholarships"]
            lines.append("=== SCHOLARSHIPS ===")
            if sch.get('schemes'):
                for scheme_id, scheme in sch['schemes'].items():
                    if isinstance(scheme, dict):
                        lines.append(f"- {scheme.get('name', scheme_id)}: {scheme.get('benefit', scheme.get('amount', ''))}")
            lines.append("")
        
        # Hostel
        if "hostel" in kb_data:
            hostel = kb_data["hostel"]
            lines.append("=== HOSTEL FACILITIES ===")
            lines.append(f"Capacity: {hostel.get('total_capacity', 1500)} students")
            if hostel.get('mess'):
                mess = hostel['mess']
                lines.append(f"Mess: Rs. {mess.get('monthly_charge', 5000)}/month")
                lines.append(f"Meals: {mess.get('meals_per_day', 4)} times/day")
                if mess.get('quality'):
                    lines.append(f"Food Quality: {mess['quality']}")
            if hostel.get('room_types'):
                lines.append("Room Types:")
                for room in hostel['room_types'][:2]:
                    if isinstance(room, dict):
                        lines.append(f"  - {room.get('type')}: Rs. {room.get('rent_per_sem')}/sem")
            if hostel.get('rules'):
                rules = hostel['rules']
                if isinstance(rules, dict):
                    lines.append("Hostel Rules:")
                    if rules.get('boys_curfew'):
                        lines.append(f"  Boys Curfew: {rules['boys_curfew']}")
                    if rules.get('girls_curfew'):
                        lines.append(f"  Girls Curfew: {rules['girls_curfew']}")
                    if rules.get('guests'):
                        lines.append(f"  Guests: {rules['guests']}")
                    if rules.get('entry_exit'):
                        lines.append(f"  Entry/Exit: {rules['entry_exit']}")
            lines.append("")
        
        # Faculty & Academics
        if "academics" in kb_data:
            ac = kb_data["academics"]
            lines.append("=== ACADEMICS ===")
            if ac.get('faculty'):
                f = ac['faculty']
                lines.append(f"Faculty: {f.get('total', '150+')}")
                if f.get('description'):
                    lines.append(f"  {f['description']}")
                lines.append(f"Student-Teacher Ratio: {ac.get('student_teacher_ratio', '15:1 to 20:1')}")
            if ac.get('teaching_methodology'):
                lines.append(f"Teaching Method: {ac['teaching_methodology']}")
            if ac.get('teaching_quality'):
                tq = ac['teaching_quality']
                if isinstance(tq, dict):
                    if tq.get('approach'):
                        lines.append(f"Approach: {tq['approach']}")
                    if tq.get('methods'):
                        methods = tq['methods']
                        if isinstance(methods, list):
                            lines.append(f"Methods: {', '.join(methods[:3])}")
                    if tq.get('support'):
                        lines.append(f"Support: {tq['support']}")
            if ac.get('exam_pattern'):
                lines.append(f"Exam Pattern: {ac['exam_pattern'].get('structure', 'CA + Semester')}")
            if ac.get('syllabus'):
                syl = ac['syllabus']
                if isinstance(syl, dict):
                    lines.append(f"Syllabus: {syl.get('status', 'Regularly updated')}")
                    if syl.get('authority'):
                        lines.append(f"  Authority: {syl['authority']}")
            if ac.get('practicals'):
                prac = ac['practicals']
                if isinstance(prac, dict):
                    lines.append(f"Practicals: {prac.get('conducted', 'Conducted as per calendar')}")
                    if prac.get('labs'):
                        lines.append(f"  Labs: {prac['labs']}")
            if ac.get('easiest_branch'):
                eb = ac['easiest_branch']
                if isinstance(eb, dict):
                    if eb.get('perceived_easier'):
                        lines.append(f"Easier Branches: {eb['perceived_easier']}")
            lines.append("")
        
        # Infrastructure
        if "infrastructure" in kb_data:
            inf = kb_data["infrastructure"]
            lines.append("=== INFRASTRUCTURE ===")
            if inf.get('wifi'):
                wifi = inf['wifi']
                if isinstance(wifi, dict):
                    lines.append(f"WiFi: Available ({wifi.get('coverage', 'Full campus')})")
                else:
                    lines.append("WiFi: Available (Full campus)")
            if inf.get('library'):
                lib = inf['library']
                if isinstance(lib, dict):
                    lines.append(f"Library: {lib.get('books', '80,000+')} books")
                    if lib.get('e_resources'):
                        lines.append(f"  E-resources: {', '.join(lib['e_resources'][:4])}")
            if inf.get('computer_labs'):
                labs = inf['computer_labs']
                if isinstance(labs, dict):
                    lines.append(f"Labs: {labs.get('description', 'Well-equipped')}")
                    if labs.get('software'):
                        lines.append(f"  Software: {', '.join(labs['software'][:5])}")
            if inf.get('canteen'):
                cant = inf['canteen']
                if isinstance(cant, dict):
                    lines.append(f"Canteen: {cant.get('description', 'Available')}")
            if inf.get('sports'):
                sports = inf['sports']
                if isinstance(sports, dict):
                    all_sports = []
                    if sports.get('outdoor'):
                        all_sports.extend(sports['outdoor'][:3])
                    if sports.get('indoor'):
                        all_sports.extend(sports['indoor'][:3])
                    if all_sports:
                        lines.append(f"Sports: {', '.join(all_sports)}")
            if inf.get('medical'):
                med = inf['medical']
                if isinstance(med, dict):
                    lines.append(f"Medical: {med.get('emergency', '24/7 emergency services available')}")
            lines.append("")
        
        # Admission Documents
        if "admission_documents" in kb_data:
            docs = kb_data["admission_documents"]
            lines.append("=== ADMISSION DOCUMENTS ===")
            all_docs = []
            for key, values in docs.items():
                if isinstance(values, list):
                    all_docs.extend(values)
            if all_docs:
                lines.append("Required documents:")
                for doc in all_docs[:10]:
                    lines.append(f"  - {doc}")
            lines.append("")
        
        # Refund Policy
        if "fees_summary" in kb_data and "refund_policy" in kb_data["fees_summary"]:
            refund = kb_data["fees_summary"]["refund_policy"]
            lines.append("=== FEE REFUND POLICY ===")
            if isinstance(refund, dict):
                if refund.get('guidelines'):
                    lines.append(f"Guidelines: {refund['guidelines']}")
                if refund.get('within_timeline'):
                    lines.append(f"Within Timeline: {refund['within_timeline']}")
                if refund.get('post_timeline_no_replacement'):
                    lines.append(f"No Replacement: {refund['post_timeline_no_replacement']}")
            else:
                lines.append(str(refund))
            lines.append("")
        
        # Anti-Ragging
        if "anti_ragging" in kb_data:
            ar = kb_data["anti_ragging"]
            lines.append("=== ANTI-RAGGING ===")
            if isinstance(ar, dict):
                lines.append(f"Policy: {ar.get('policy', 'Zero Tolerance')}")
                if ar.get('measures'):
                    lines.append("Measures:")
                    for m in ar['measures'][:3]:
                        lines.append(f"  - {m}")
                if ar.get('safety'):
                    lines.append(f"Safety: {ar['safety']}")
            lines.append("")
        
        # Branch Change
        if "branch_change" in kb_data:
            bc = kb_data["branch_change"]
            lines.append("=== BRANCH CHANGE ===")
            if isinstance(bc, dict):
                lines.append(f"Allowed: {bc.get('allowed', 'Yes')}")
                if bc.get('timing'):
                    lines.append(f"Timing: {bc['timing']}")
                if bc.get('criteria'):
                    lines.append(f"Criteria: {bc['criteria']}")
                if bc.get('authority'):
                    lines.append(f"Authority: {bc['authority']}")
            lines.append("")
        
        # Internships
        if "placements" in kb_data and "internship" in kb_data["placements"]:
            intern = kb_data["placements"]["internship"]
            lines.append("=== INTERNSHIPS ===")
            if isinstance(intern, dict):
                if intern.get('available'):
                    lines.append(f"Available: Yes")
                if intern.get('description'):
                    lines.append(f"{intern['description']}")
                if intern.get('top_partners'):
                    lines.append(f"Partners: {', '.join(intern['top_partners'][:5])}")
                if intern.get('software_stipend'):
                    lines.append(f"Software Stipend: {intern['software_stipend']}")
            lines.append("")
        
        # Training Cell
        if "placements" in kb_data and "training_cell" in kb_data["placements"]:
            tc = kb_data["placements"]["training_cell"]
            lines.append("=== PLACEMENT TRAINING ===")
            if isinstance(tc, dict):
                if tc.get('programs'):
                    lines.append("Programs:")
                    for prog, desc in tc['programs'].items():
                        lines.append(f"  - {prog.title()}: {desc}")
            lines.append("")
        
        # Student Life
        if "student_life" in kb_data:
            sl = kb_data["student_life"]
            lines.append("=== CAMPUS LIFE ===")
            if sl.get('tech_fest'):
                tf = sl['tech_fest']
                if isinstance(tf, dict):
                    lines.append(f"Tech Fest: {tf.get('name', 'HORIZON')} ({tf.get('month', 'February')})")
            if sl.get('cultural_fest'):
                cf = sl['cultural_fest']
                if isinstance(cf, dict):
                    lines.append(f"Cultural Fest: {cf.get('name', 'ZEAL')} ({cf.get('month', 'January')})")
            if sl.get('clubs'):
                lines.append(f"Clubs: {', '.join(sl['clubs'][:5])}")
            if sl.get('description'):
                lines.append(f"Overall: {sl['description']}")
            lines.append("")
        
        # Departments
        if "departments" in kb_data:
            dept = kb_data["departments"]
            lines.append("=== DEPARTMENTS ===")
            for code, info in list(dept.items())[:15]:
                if isinstance(info, dict):
                    hod_name = info.get('hod', {}).get('name', 'N/A')
                    lines.append(f"{code}: HOD - {hod_name}")
            lines.append("")
        
        # Principal & Vice Principal
        if "principal" in kb_data or "vice_principal" in kb_data:
            lines.append("=== COLLEGE LEADERSHIP ===")
            if "principal" in kb_data:
                p = kb_data["principal"]
                lines.append(f"Principal: {p.get('name', 'Dr. Sanjay S. Pawar')}")
                if p.get('email'):
                    lines.append(f"  Email: {p.get('email')}")
            if "vice_principal" in kb_data:
                vp = kb_data["vice_principal"]
                lines.append(f"Vice Principal: {vp.get('name', 'Prof. (Dr.) K. M. Hossain')}")
                if vp.get('email'):
                    lines.append(f"  Email: {vp.get('email')}")
            lines.append("")
        
        # Quick Answers
        if "quick_answers" in kb_data:
            qa = kb_data["quick_answers"]
            lines.append("=== QUICK FACTS ===")
            for key, value in list(qa.items())[:5]:
                lines.append(f"- {key.replace('_', ' ').title()}: {value}")
            lines.append("")
        
        # Important Links
        if "important_links" in kb_data:
            links = kb_data["important_links"]
            lines.append("=== IMPORTANT LINKS ===")
            if links.get('website'):
                lines.append(f"Website: {links['website']}")
            if links.get('admission_portal'):
                lines.append(f"Admission Portal: {links['admission_portal']}")
            if links.get('placements'):
                lines.append(f"Placements: {links['placements']}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_conversation_history(self, history: List[Dict]) -> str:
        """Format conversation history for the prompt"""
        if not history:
            return ""
        
        lines = ["=== CONVERSATION HISTORY ==="]
        for msg in history[-4:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")[:200]
            lines.append(f"{role}: {content}")
        lines.append("")
        return "\n".join(lines)
    
    async def generate_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using Groq.
        """
        if not self.client:
            return {"answer": "API Error", "source": "error", "model": None}
        
        try:
            # 1. Enhanced Script & Language Detection
            query_lower = query.lower()
            
            # Detect Bengali
            is_bengali = (
                any('\u0980' <= c <= '\u09FF' for c in query) or 
                any(word in query_lower for word in ["koto", "ki ", "btech e", "cse te", "bolun", "hobe", "fiss", "fees koto"])
            )
            
            # Detect Hindi (if not Bengali)
            is_hindi = False
            if not is_bengali:
                is_hindi = (
                    any('\u0900' <= c <= '\u097F' for c in query) or
                    any(word in query_lower for word in ["kya ", "hai ", "kaise", "fees kitni", "bataiye", "kab ", "admission process"])
                )
            
            lang_instruction = "Respond ONLY in English."
            if is_bengali:
                lang_instruction = "Respond ONLY in Bengali Script (বাংলা). Use 'টাকা' for currency and Bengali digits."
            elif is_hindi:
                lang_instruction = "Respond ONLY in Hindi Script (हिन्दी). Use standard digits (0-9)."

            history_context = self._format_conversation_history(conversation_history) if conversation_history else ""
            
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"KNOWLEDGE BASE:\n{self.knowledge_base}\n\n{history_context}\nUser Query: {query}\n\nSTRICT: {lang_instruction}. START WITH THE ANSWER. MAXIMUM 2 SENTENCES."}
                ],
                temperature=0.1,
                max_tokens=self.max_tokens
            )
            
            answer = chat_completion.choices[0].message.content.strip()
            
            # Post-processing for Bengali Currency leak
            if is_bengali:
                answer = answer.replace("রুপি", "টাকা").replace("টাকা.", "টাকা।")
            else:
                # Remove Bengali digits if they leaked into non-Bengali response
                bengali_to_std = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
                answer = answer.translate(bengali_to_std)
            
            # We rely purely on the prompt and max_tokens to enforce brevity,
            # as python regex truncation aggressively breaks on abbreviations like 'Dr. B. C. Roy'.
            
            # Generate a separate version specifically for the Voice engine
            voice_text = self._clean_for_voice(answer)
            
            logger.info(f"Groq response generated: {len(answer)} chars")
            
            return {
                "answer": answer,
                "voice_text": voice_text,
                "source": "groq",
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"Groq generation error: {e}")
            return {
                "answer": f"I encountered an error processing your request. Please try again or contact the college at 0343-2501353.",
                "voice_text": "I encountered an error. Please try again or call our support.",
                "source": "error",
                "error": str(e)
            }

    def _clean_for_voice(self, text: str) -> str:
        """Safety net to force phonetic dots and number words for natural TTS"""
        import re
        
        # 1. Force dots in acronyms (even if LLM forgets)
        # Only targets acronyms surrounded by whitespace/punctuation — NOT inside emails or URLs
        acronyms = ["CSE", "AIML", "IT", "ECE", "EE", "ME", "CE", "WBJEE", "MBA", "MCA", "BTECH"]
        clean = text
        for ac in acronyms:
            # Negative lookbehind/lookahead: skip if preceded or followed by @ . / (email/URL context)
            pattern = r'(?<![@.\w])' + ac + r'(?![.\w@/])'
            phonetic = " ".join(list(ac))
            clean = re.sub(pattern, phonetic, clean, flags=re.IGNORECASE)
            
        # 2. Convert common digits to words (to prevent robotic TTS)
        # Handles both 5,98,300 and 598,300 formats
        replacements = [
            ("5,98,300", "five lakh ninety-eight thousand three hundred"),
            ("598,300", "five lakh ninety-eight thousand three hundred"),
            ("5,47,700", "five lakh forty-seven thousand seven hundred"),
            ("547,700", "five lakh forty-seven thousand seven hundred"),
            ("4,37,700", "four lakh thirty-seven thousand seven hundred"),
            ("437,700", "four lakh thirty-seven thousand seven hundred"),
            ("97,125", "ninety-seven thousand one hundred twenty-five"),
            ("80,500", "eighty thousand five hundred"),
            ("5,000", "five thousand"),
            ("30", "thirty"),
            ("15", "fifteen"),
            ("85%", "eighty-five percent"),
            ("80.62%", "eighty point six two percent"),
        ]
        
        # Only apply number replacements if text is English
        if not re.search(r"[\u0900-\u09FF]", clean):
            # Sort replacements by length (longest first) to avoid partial matches
            for old, new in sorted(replacements, key=lambda x: len(x[0]), reverse=True):
                # Use word boundaries for small numbers to avoid matching inside larger ones
                if len(old) <= 2:
                    clean = re.sub(r'\b' + old + r'\b', new, clean)
                else:
                    clean = clean.replace(old, new)
            
        return clean
    
    def is_available(self) -> bool:
        """Check if Groq service is available"""
        return self.client is not None

    def get_voice_kb(self) -> str:
        """Get a natural language version of the KB for flawless TTS pronunciation"""
        return self.knowledge_base


# Global instance
_groq_service = None


def get_groq_service() -> GroqService:
    """Get or create the Groq service singleton"""
    global _groq_service
    if _groq_service is None:
        _groq_service = GroqService()
    return _groq_service
