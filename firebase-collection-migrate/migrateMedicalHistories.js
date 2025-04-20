const admin = require('firebase-admin');
const serviceAccount = require('../Cred/Firebase/Med App/med-labs-42f13-firebase-adminsdk-fbsvc-43b78dfef7.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

const db = admin.firestore();

async function migrateMedicalHistories() {
  try {
    const medicalHistoriesRef = db.collection('medical_histories');
    const snapshot = await medicalHistoriesRef.get();

    if (snapshot.empty) {
      console.log('No documents to migrate in medical_histories.');
      return;
    }

    const batch = db.batch();

    for (const doc of snapshot.docs) {
      const docId = doc.id;
      if (docId.startsWith('history_')) {
        const newDocId = docId.replace('history_', '');
        const data = doc.data();

        // Create new document with the new ID
        const newDocRef = medicalHistoriesRef.doc(newDocId);
        batch.set(newDocRef, {
          uid: data.uid,
          summary: data.summary,
          timestamp: data.timestamp
        });

        // Add to summaries subcollection
        const summaryDocRef = newDocRef.collection('summaries').doc();
        batch.set(summaryDocRef, {
          summary: data.summary,
          timestamp: data.timestamp
        });

        // Delete the old document
        batch.delete(doc.ref);

        console.log(`Migrating ${docId} to ${newDocId}`);
      }
    }

    await batch.commit();
    console.log('Migration completed successfully.');
  } catch (error) {
    console.error('Error during migration:', error);
  } finally {
    process.exit();
  }
}

migrateMedicalHistories();