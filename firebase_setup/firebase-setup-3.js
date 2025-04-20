/**
 * firebase_firestore_data_setup.js
 *
 * This script demonstrates how to add prescription and lab records to Firestore,
 * keyed by patient UID, with doctor and assistant IDs, and GCS bucket links.
 *
 * NOTE: This script assumes you have Firebase Admin SDK initialized and
 * have the necessary permissions to write to Firestore.
 */

const admin = require('firebase-admin');

// Initialize Firebase Admin SDK (replace with your service account credentials)
const serviceAccount = require('./Cred/Firebase/Med App/med-labs-42f13-firebase-adminsdk-fbsvc-43b78dfef7.json'); // Path to your service account key

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

const db = admin.firestore();

async function addPrescriptionRecord(patientId, doctorIds, gcsLink) {
  try {
    const prescriptionRef = db.collection('prescriptions').doc(patientId);
    await prescriptionRef.set({
      patient_id: patientId,
      doctor_ids: doctorIds, // Array of doctor IDs
      gcs_link: gcsLink,
      timestamp: admin.firestore.FieldValue.serverTimestamp()
    });
    console.log(`Prescription record added for patient ${patientId}`);
  } catch (error) {
    console.error(`Error adding prescription record: ${error}`);
  }
}

async function addLabRecord(patientId, doctorIds, assistantIds, gcsLink) {
  try {
    const labRef = db.collection('lab_records').doc(patientId);
    await labRef.set({
      patient_id: patientId,
      doctor_ids: doctorIds, // Array of doctor IDs
      assistant_ids: assistantIds, // Array of assistant IDs
      gcs_link: gcsLink,
      timestamp: admin.firestore.FieldValue.serverTimestamp()
    });
    console.log(`Lab record added for patient ${patientId}`);
  } catch (error) {
    console.error(`Error adding lab record: ${error}`);
  }
}

async function main() {
  // Example patient UIDs and data
  const patient1Uid = 'patient123';
  const patient2Uid = 'patient456';

  // Example doctor and assistant IDs (replace with actual IDs from consultant_registrations)
  const doctor1Id = 'doctor001';
  const doctor2Id = 'doctor002';
  const assistant1Id = 'assistant101';
  const assistant2Id = 'assistant102';

  // Example GCS bucket links
  const prescriptionLink1 = 'gs://your-bucket/prescriptions/patient123.pdf';
  const prescriptionLink2 = 'gs://your-bucket/prescriptions/patient456.pdf';
  const labLink1 = 'gs://your-bucket/lab_reports/patient123.jpg';
  const labLink2 = 'gs://your-bucket/lab_reports/patient456.png';

  // Add prescription records
  await addPrescriptionRecord(patient1Uid, [doctor1Id], prescriptionLink1);
  await addPrescriptionRecord(patient2Uid, [doctor2Id, doctor1Id], prescriptionLink2);

  // Add lab records
  await addLabRecord(patient1Uid, [doctor1Id], [assistant1Id], labLink1);
  await addLabRecord(patient2Uid, [doctor2Id, doctor1Id], [assistant2Id, assistant1Id], labLink2);

  console.log('Data setup complete.');
}

main();