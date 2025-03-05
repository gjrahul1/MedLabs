const admin = require('firebase-admin');
const serviceAccount = require('../Cred/Firebase/Med App/med-labs-42f13-firebase-adminsdk-fbsvc-43b78dfef7.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  databaseURL: "https://med-labs-42f13.firebaseio.com"
});

const db = admin.firestore();

async function migrateConsultantIds() {
  const snapshot = await db.collection('consultant_registrations').where('role', '==', 'doctor').get();
  let maxNumber = 0;

  for (const doc of snapshot.docs) {
    const data = doc.data();
    if (!data.consultant_id) {
      maxNumber++;
      const newId = `DR00${maxNumber.toString().padStart(2, '0')}`;
      data.consultant_id = newId;
      data.doctor_id = newId;

      // Update the document with new ID
      const newDocRef = db.collection('consultant_registrations').doc(newId);
      await newDocRef.set(data);
      await doc.ref.delete();
      console.log(`Migrated ${doc.id} to ${newId}`);

      // Update prescriptions
      const prescriptions = await db.collection('prescriptions').where('consultant_id', '==', doc.id).get();
      for (const presDoc of prescriptions.docs) {
        await presDoc.ref.update({ consultant_id: newId });
        console.log(`Updated prescription ${presDoc.id} with new consultant_id ${newId}`);
      }
    }
  }
}

migrateConsultantIds().then(() => console.log('Migration complete')).catch(console.error);