// Import and initialize Firebase Admin SDK
const admin = require('firebase-admin');

const serviceAccount = require('./Cred/Firebase/Med App/med-labs-42f13-firebase-adminsdk-fbsvc-43b78dfef7.json'); // Ensure correct path

// Initialize Firebase Admin
if (!admin.apps.length) {
  admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
  });
}

const db = admin.firestore();

// Function to ensure patient_counter document exists
async function setupMetadata() {
  const counterRef = db.collection('metadata').doc('patient_counter');

  try {
    const docSnapshot = await counterRef.get();

    if (!docSnapshot.exists) {
      console.log('Creating missing patient_counter document...');
      await counterRef.set({ count: 0 });
      console.log('✅ patient_counter initialized to 0.');
    } else {
      console.log('✅ patient_counter already exists:', docSnapshot.data());
    }
  } catch (error) {
    console.error('❌ Error setting up metadata:', error);
  }
}

// Execute the setup
setupMetadata();
