"""Generate SYNTHETIC candidate profiles (no real people)."""
import random, pandas as pd
random.seed(7)
SPECS = {"ICU": ["critical care", "ventilator management", "hemodynamic monitoring"],
         "ER": ["emergency", "triage", "trauma care"],
         "Med-Surg": ["medical surgical", "patient assessment", "wound care"],
         "OR": ["operating room", "scrub nurse", "sterile technique"],
         "Telemetry": ["cardiac monitoring", "ECG interpretation", "telemetry"],
         "Pediatrics": ["pediatric care", "family communication", "immunization"]}
ROLES = ["RN", "RN", "RN", "LPN", "Respiratory Therapist", "Physical Therapist"]
CERTS = ["BLS", "ACLS", "PALS", "NIHSS", "TNCC"]
SHIFTS = ["day", "night", "rotating", "flexible"]
FIRST = ["Asha", "Ben", "Carla", "Dev", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jon", "Kavya", "Liam"]
rows = []
for i in range(200):
    spec = random.choice(list(SPECS)); role = random.choice(ROLES)
    certs = random.sample(CERTS, random.randint(1, 4))
    yrs = random.randint(0, 15); shift = random.choice(SHIFTS)
    summary = (f"{role} with {yrs} years in {spec} care. Skills: {', '.join(random.sample(SPECS[spec], 2))}. "
               f"Certifications: {', '.join(certs)}. Open to {shift} shift. Licensed in multiple states.")
    rows.append({"candidate_id": f"P{i+1:03d}", "name": f"{random.choice(FIRST)} #{i+1}",
                 "role": role, "specialty": spec, "years_exp": yrs, "certs": "|".join(certs),
                 "shift": shift, "summary": summary})
pd.DataFrame(rows).to_csv("candidates_synthetic.csv", index=False)
print("Wrote 200 synthetic candidates")
