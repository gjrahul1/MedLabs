// registerAssistantsConsultants.js

const { initializeApp, cert } = require('firebase-admin/app');
const { getFirestore } = require('firebase-admin/firestore');

// Path to your Firebase Admin SDK JSON file
const serviceAccount = require('./Cred/Firebase/Med App/med-labs-42f13-firebase-adminsdk-fbsvc-43b78dfef7.json');

// Initialize Firebase Admin
initializeApp({
  credential: cert(serviceAccount),
});

const db = getFirestore();

// Helper function to get the current timestamp
const getTimestamp = () => new Date();

// Function to register assistants
async function registerAssistants() {
  try {
    const assistants = [
      {
        assistant_id: 'A-202501',
        lab_id: 'LAB001',
        full_name: 'Jane Doe',
        email: 'jane.doe@example.com',
        phone: '+1234567891',
        assigned_doctors: ['DR001'],
        role: 'assistant',
        created_at: getTimestamp(),
      },
      {
        assistant_id: 'A-202502',
        lab_id: 'LAB002',
        full_name: 'Alice Brown',
        email: 'alice.brown@example.com',
        phone: '+9876543210',
        assigned_doctors: ['DR002'],
        role: 'assistant',
        created_at: getTimestamp(),
      },
    ];

    for (const assistant of assistants) {
      await db.collection('assistant_registrations')
        .doc(assistant.assistant_id)
        .set(assistant);
      console.log(`✅ Assistant Registered: ${assistant.full_name}`);
    }
  } catch (error) {
    console.error('❌ Error registering assistants:', error);
  }
}

// Function to register consultants
async function registerConsultants() {
  try {
    const consultants = [
      {
        doctor_id: 'DR001',
        full_name: 'Dr. John Smith',
        email: 'dr.john@example.com',
        phone: '+1234567890',
        department: 'Cardiology',
        patients: ['P-654321', 'P-123456'],
        role: 'doctor',
        created_at: getTimestamp(),
      },
      {
        doctor_id: 'DR002',
        full_name: 'Dr. Emma Johnson',
        email: 'emma.johnson@example.com',
        phone: '+1122334455',
        department: 'Neurology',
        patients: ['P-789012'],
        role: 'doctor',
        created_at: getTimestamp(),
      },
    ];

    for (const consultant of consultants) {
      await db.collection('consultant_registrations')
        .doc(consultant.doctor_id)
        .set(consultant);
      console.log(`✅ Consultant Registered: ${consultant.full_name}`);
    }
  } catch (error) {
    console.error('❌ Error registering consultants:', error);
  }
}

// Main function to run both registrations
async function main() {
  await registerAssistants();
  await registerConsultants();
  console.log('🎉 Registration completed successfully.');
}

main();
