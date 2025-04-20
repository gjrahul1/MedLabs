// modify_consultant_registrations.js

const admin = require('firebase-admin');
const serviceAccount = require('../Cred/Firebase/Med App/med-labs-42f13-firebase-adminsdk-fbsvc-43b78dfef7.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  databaseURL: 'https://medical-app-56e1e.firebaseio.com'
});

const db = admin.firestore();
const auth = admin.auth();

(async () => {
  const doctors = [
    { email: "dr.ravi.derma@gmail.com", full_name: "Dr. Ravi Deshmukh", phone: "9012345678", specialty: "Dermatology" },
    { email: "dr.meera.ent@gmail.com", full_name: "Dr. Meera Kapoor", phone: "9123456780", specialty: "ENT" },
    { email: "dr.sanjay.nephro@gmail.com", full_name: "Dr. Sanjay Rao", phone: "9234567810", specialty: "Nephrology" },
    { email: "dr.latha.endo@gmail.com", full_name: "Dr. Latha Nair", phone: "9345678120", specialty: "Endocrinology" },
    { email: "dr.ashok.gastro@gmail.com", full_name: "Dr. Ashok Iyer", phone: "9456781230", specialty: "Gastroenterology" },
    { email: "dr.nisha.uro@gmail.com", full_name: "Dr. Nisha Bhatt", phone: "9567812340", specialty: "Urology" },
    { email: "dr.karan.psy@gmail.com", full_name: "Dr. Karan Malhotra", phone: "9678123450", specialty: "Psychiatry" },
    { email: "dr.alka.gen@gmail.com", full_name: "Dr. Alka Verma", phone: "9781234560", specialty: "General Medicine" },
    { email: "dr.rajesh.surg@gmail.com", full_name: "Dr. Rajesh Rana", phone: "9892345670", specialty: "General Surgery" },
    { email: "dr.shalini.gyne@gmail.com", full_name: "Dr. Shalini Joshi", phone: "9903456781", specialty: "Gynecology" },
    { email: "dr.vikas.onco@gmail.com", full_name: "Dr. Vikas Sharma", phone: "9812345671", specialty: "Oncology" },
    { email: "dr.nandita.pedia@gmail.com", full_name: "Dr. Nandita Rao", phone: "9823456782", specialty: "Pediatrics" },
    { email: "dr.manoj.cardio@gmail.com", full_name: "Dr. Manoj Mehta", phone: "9834567893", specialty: "Cardiology" },
    { email: "dr.supriya.neuro@gmail.com", full_name: "Dr. Supriya Sen", phone: "9845678914", specialty: "Neurology" },
    { email: "dr.anand.ortho@gmail.com", full_name: "Dr. Anand Kulkarni", phone: "9856789125", specialty: "Orthopedics" }
  ];

  try {
    for (let i = 0; i < doctors.length; i++) {
      const doctor = doctors[i];
      const consultant_id = `DR${(i + 6).toString().padStart(4, '0')}`; // Starts from DR0006
      const uid = (await auth.createUser({
        email: doctor.email,
        password: "tempPassword123", // Set a default password; update later if needed
        displayName: doctor.full_name
      })).uid;

      const doctorData = {
        consultant_id,
        doctor_id: consultant_id,
        email: doctor.email,
        full_name: doctor.full_name,
        phone: doctor.phone,
        role: "doctor",
        specialty: doctor.specialty,
        uid,
        availability: true,
        assigned_patients: []
      };

      const docRef = db.collection('consultant_registrations').doc(consultant_id);
      const docSnap = await docRef.get();
      if (!docSnap.exists) {
        await docRef.set(doctorData);
        console.log(`Doctor added: ${doctor.full_name} with consultant_id ${consultant_id} and UID ${uid}`);
      } else {
        console.log(`Doctor with consultant_id ${consultant_id} already exists. Skipping.`);
      }
    }
    console.log("All doctor registrations completed successfully!");
  } catch (error) {
    console.error("Error adding doctors: ", error.message);
    process.exit(1); // Exit with error code
  }
})();