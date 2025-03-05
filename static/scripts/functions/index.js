const functions = require('firebase-functions');
const admin = require('firebase-admin');
admin.initializeApp();

exports.setRoleOnUserCreate = functions.auth.user().onCreate(async (user) => {
    let role = null;
    let collectionName = null;

    try {
        // Attempt to find user in patient_registrations
        const patientDoc = await admin.firestore().doc(`patient_registrations/${user.uid}`).get();
        if (patientDoc.exists) {
            role = patientDoc.data().role;
            collectionName = 'patient_registrations';
        }

        // Attempt to find user in consultant_registrations
        if (!role) {
            const consultantDoc = await admin.firestore().doc(`consultant_registrations/${user.uid}`).get();
            if (consultantDoc.exists) {
                role = consultantDoc.data().role;
                collectionName = 'consultant_registrations';
            }
        }

        // Attempt to find user in assistant_registrations
        if (!role) {
            const assistantDoc = await admin.firestore().doc(`assistant_registrations/${user.uid}`).get();
            if (assistantDoc.exists) {
                role = assistantDoc.data().role;
                collectionName = 'assistant_registrations';
            }
        }

        if (role) {
            await admin.auth().setCustomUserClaims(user.uid, { role });
            console.log(`Role '${role}' set for user ${user.uid} from collection ${collectionName}`);
        } else {
            console.error(`Role not found for user ${user.uid}`);
        }
    } catch (error) {
        console.error('Error setting custom claim:', error);
    }
});