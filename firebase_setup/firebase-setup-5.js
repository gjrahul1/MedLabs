// create_initial_screenings.js

const admin = require('firebase-admin');
const serviceAccount = require('../Cred/Firebase/Med App/med-labs-42f13-firebase-adminsdk-fbsvc-43b78dfef7.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  databaseURL: 'https://medical-app-56e1e.firebaseio.com'
});

const db = admin.firestore();

(async () => {
  try {
    const screeningData = {
      uid: "QFIq09sMHeHdiAXdGZC8xmZw6i1", // Replace with actual patient UID
      consultant_id: "DR0003",
      timestamp: admin.firestore.FieldValue.serverTimestamp(),
      language: "ta",
      medical_data: {
        symptoms: "Fever",
        severity: "Moderate",
        duration: "3 days",
        triggers: null
      },
      patient_name: "John Doe",
      status: "pending"
    };

    const docRef = db.collection('initial_screenings').doc();
    await docRef.set(screeningData);
    console.log(`Initial screening created with doc_id: ${docRef.id}`);
  } catch (error) {
    console.error("Error creating initial screenings: ", error);
  }
})();