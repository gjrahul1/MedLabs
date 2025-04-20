// firebase-data-setup.js
const { initializeApp, cert } = require('firebase-admin/app');
const { getFirestore } = require('firebase-admin/firestore');
const { Storage } = require('@google-cloud/storage');
const path = require('path');
const fs = require('fs');

// Replace with your service account key file path
const serviceAccount = require('./Cred/Firebase/Med App/med-labs-42f13-firebase-adminsdk-fbsvc-43b78dfef7.json');

initializeApp({
  credential: cert(serviceAccount),
  storageBucket: 'med-labs-42f13.appspot.com',
});

const db = getFirestore();
const storage = new Storage({
  projectId: serviceAccount.project_id,
  keyFilename: './Cred/Firebase/Med App/med-labs-42f13-firebase-adminsdk-fbsvc-43b78dfef7.json',
});
const bucket = storage.bucket('med-labs-42f13');

async function uploadPdf(patientId, type, filePath, fileName) {
  try {
    const destination = `patients/${patientId}/${type}s/${fileName}`;
    await bucket.upload(filePath, {
      destination: destination,
    });

    const fileUrl = `gs://${bucket.name}/${destination}`;
    return fileUrl;
  } catch (error) {
    console.error('Error uploading PDF:', error);
    return null;
  }
}

async function addPdfData(patient_id, type, filePath, fileName) {
  const fileUrl = await uploadPdf(patient_id, type, filePath, fileName);
  if (fileUrl) {
    await db.collection('pdf_files').add({
      patient_id: patient_id,
      type: type,
      file_url: fileUrl,
      uploaded_at: new Date(),
    });
    console.log('PDF data added to Firestore');
  }
}

async function setupData(pdfFilePath) {
  try {
    // 1. patient_registrations Collection (already added)
    await db.collection('patient_registrations').doc('john.doe@example.com').set({
      patient_id: 'P-123456',
      full_name: 'John Doe',
      email: 'john.doe@example.com',
      phone: '+1234567890',
      dob: '1990-05-22',
      location: 'New York, USA',
      role: 'patient',
      consultant_id: 'DR001',
      disease: 'Hypertension',
      last_visit: new Date('2025-02-27T00:00:00Z'),
      prescription_summary: 'Blood pressure medication prescribed.',
      lab_summary: 'Cholesterol levels normal.',
    });
    await db.collection('patient_registrations').doc('jane.doe@example.com').set({
      patient_id: 'P-654321',
      full_name: 'Jane Doe',
      email: 'jane.doe@example.com',
      phone: '+1234567891',
      dob: '1991-06-23',
      location: 'London, UK',
      role: 'patient',
      consultant_id: 'DR001',
      disease: 'Diabetes',
      last_visit: new Date('2025-02-26T00:00:00Z'),
      prescription_summary: 'Insulin prescribed.',
      lab_summary: 'Blood sugar levels high.',
    });

    // 2. consultants Collection (add new data)
    await db.collection('consultants').doc('DR001').set({
      doctor_id: 'DR001',
      full_name: 'Dr. John Smith',
      email: 'dr.john@example.com',
      phone: '+1234567890',
      department: 'Cardiology',
      patients: ['P-654321', 'P-123456'],
      role: 'doctor',
    });

    // 3. pdf_files Collection (already added)
    const fileName = path.basename(pdfFilePath);
    await addPdfData('P-123456', 'prescription', pdfFilePath, fileName);

    // 4. assistants Collection (add new data)
    await db.collection('assistants').doc('A-202501').set({
      assistant_id: 'A-202501',
      lab_id: 'LAB001',
      full_name: 'Jane Doe',
      email: 'jane.doe@example.com',
      phone: '+1234567891',
      assigned_doctors: ['DR001'],
      role: 'assistant',
    });

    console.log('Data setup complete.');
  } catch (error) {
    console.error('Error setting up data:', error);
  }
}

// Get the file path from command-line arguments
const pdfFilePath = process.argv[2];

if (pdfFilePath) {
  setupData(pdfFilePath);
} else {
  console.error('PDF file path not provided as a command-line argument.');
}