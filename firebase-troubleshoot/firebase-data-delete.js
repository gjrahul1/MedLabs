const { initializeApp, cert } = require('firebase-admin/app');
const { getFirestore } = require('firebase-admin/firestore');

// Path to your Firebase Admin SDK JSON file (same as provided script)
const serviceAccount = require('../Cred/Firebase/Med App/med-labs-42f13-firebase-adminsdk-fbsvc-43b78dfef7.json');

// Initialize Firebase Admin
initializeApp({
  credential: cert(serviceAccount),
});

const db = getFirestore();

// List of collections to clear
const collectionsToClear = [
  'initial_screenings',
  'lab_records',
  'medical_histories',
  'patient_registrations',
  'prescriptions'
];

// Function to delete all documents in a single collection
async function deleteCollectionData(collectionName) {
  try {
    const collectionRef = db.collection(collectionName);
    const snapshot = await collectionRef.get();

    if (snapshot.empty) {
      console.log(`⚠️ No documents found in ${collectionName}.`);
      return;
    }

    // Batch delete for efficiency
    const batch = db.batch();
    snapshot.docs.forEach(doc => {
      batch.delete(doc.ref);
    });

    await batch.commit();
    console.log(`🗑️ Deleted all documents in ${collectionName}.`);
  } catch (error) {
    console.error(`❌ Error deleting documents in ${collectionName}:`, error);
  }
}

// Main function to clear all specified collections
async function clearAllCollections() {
  for (const collection of collectionsToClear) {
    await deleteCollectionData(collection);
  }
  console.log('🎉 All specified collections have been cleared of data.');
}

// Run the main function
clearAllCollections();