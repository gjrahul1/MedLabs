//Code to reset the password of all the consultants

const admin = require('firebase-admin');

// Initialize Firebase Admin SDK
const serviceAccount = require('../Cred/Firebase/Med App/med-labs-42f13-firebase-adminsdk-fbsvc-43b78dfef7.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

// Firestore database instance
const db = admin.firestore();

async function resetConsultantPasswords() {
  try {
    console.log('Fetching consultants from consultant_registrations...');
    const snapshot = await db.collection('consultant_registrations').get();

    if (snapshot.empty) {
      console.log('No consultants found in consultant_registrations.');
      return;
    }

    console.log(`Found ${snapshot.size} consultants. Updating passwords...`);

    // Process each consultant
    const updatePromises = [];
    let successCount = 0;
    let failureCount = 0;

    for (const doc of snapshot.docs) {
      const consultant = doc.data();
      // Use the 'uid' or 'firebase_id' field instead of doc.id
      const uid = consultant.uid || consultant.firebase_id;
      const email = consultant.email;

      if (!uid || !email) {
        console.error(`Consultant with doc ID ${doc.id} is missing UID or email:`, consultant);
        failureCount++;
        continue;
      }

      // Create a promise to update the password
      const updatePromise = admin.auth().updateUser(uid, {
        password: 'tempPassword123',
      })
        .then((userRecord) => {
          console.log(`Successfully updated password for consultant ${email} (UID: ${uid})`);
          successCount++;
        })
        .catch((error) => {
          console.error(`Failed to update password for consultant ${email} (UID: ${uid}):`, error.message);
          failureCount++;
        });

      updatePromises.push(updatePromise);
    }

    // Wait for all updates to complete
    await Promise.all(updatePromises);

    console.log('\nPassword reset summary:');
    console.log(`Successfully updated passwords for ${successCount} consultants.`);
    console.log(`Failed to update passwords for ${failureCount} consultants.`);

    if (failureCount > 0) {
      console.log('Please check the logs above for details on failed updates.');
    }

  } catch (error) {
    console.error('Error during password reset process:', error);
  } finally {
    // Exit the process
    process.exit(0);
  }
}

// Run the script
resetConsultantPasswords();