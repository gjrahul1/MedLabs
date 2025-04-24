document.addEventListener('DOMContentLoaded', () => {
  console.log('Registeration.js loaded');
  console.log('Firebase initialized:', firebase.auth(), firebase.firestore());

  const registerForm = document.getElementById('registration-form');
  if (!registerForm) {
    console.error("Register form not found");
    return;
  }

  const toggleOptions = document.querySelectorAll('.toggle-option');
  const roleFields = {
    'patient': document.getElementById('patient-fields'),
    'doctor': document.getElementById('doctor-fields'),
    'assistant': document.getElementById('assistant-fields')
  };

  Object.values(roleFields).forEach((field, index) => {
    if (!field) {
      console.error(`Role field for index ${index} not found in DOM`);
      return;
    }
  });

  function showFields(role) {
    Object.keys(roleFields).forEach(r => {
      const field = roleFields[r];
      field.style.display = r === role ? 'block' : 'none';
      field.querySelectorAll('input[required], select[required]').forEach(input => {
        input.required = (r === role);
      });
    });
    document.getElementById('role').value = role;
    toggleOptions.forEach(option => option.classList.toggle('active', option.getAttribute('data-role') === role));
  }

  showFields('patient');

  toggleOptions.forEach(option => {
    option.addEventListener('click', () => {
      const role = option.getAttribute('data-role');
      showFields(role);
    });
  });

  registerForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    console.log('Registration form submitted');

    const role = document.getElementById('role').value;
    const emailField = role === 'patient' ? 'email_patient' : role === 'doctor' ? 'email_doctor' : 'email_assistant';
    const passwordField = role === 'patient' ? 'password_patient' : role === 'doctor' ? 'password_doctor' : 'password_assistant';
    const email = document.getElementById(emailField).value.trim();
    const password = document.getElementById(passwordField).value.trim();

    let userData = {
      email: email,
      password: password,
      role: role
    };

    if (role === 'patient') {
      userData.full_name = document.getElementById('full_name_patient').value.trim();
      userData.phone = document.getElementById('phone_patient').value.trim();
      userData.dob = document.getElementById('dob_patient').value.trim();
      userData.location = document.getElementById('location_patient').value.trim();
      userData.age = document.getElementById('age_patient').value.trim();
    } else if (role === 'doctor') {
      userData.full_name = document.getElementById('full_name_doctor').value.trim();
      userData.phone = document.getElementById('phone_doctor').value.trim();
      userData.specialty = document.getElementById('specialty_doctor').value.trim();
      userData.department = document.getElementById('department_doctor')?.value.trim() || 'General'; // Optional field
    } else if (role === 'assistant') {
      userData.full_name = document.getElementById('full_name_assistant').value.trim();
      userData.phone = document.getElementById('phone_assistant').value.trim();
      userData.department = document.getElementById('department_assistant').value.trim();
      const assigned_doctors = document.getElementById('assigned_doctors_assistant').value.split(',').map(d => d.trim()).filter(d => d.match(/^DR00\d+$/));
      if (assigned_doctors.length) userData.assigned_doctors = assigned_doctors;
    }

    const requiredFields = {
      'patient': ['full_name', 'email', 'password', 'phone', 'dob', 'location', 'age'],
      'doctor': ['full_name', 'email', 'password', 'phone', 'specialty'],
      'assistant': ['full_name', 'email', 'password', 'phone', 'department']
    };

    const missingFields = requiredFields[role].filter(field => !userData[field] && field !== 'assigned_doctors');
    if (missingFields.length > 0) {
      alert(`Please fill in all required fields: ${missingFields.join(', ')}`);
      return;
    }

    if (!userData.email.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) {
      alert("Please enter a valid email address.");
      return;
    }
    if (!/^\d{10}$/.test(userData.phone)) {
      alert("Please enter a valid 10-digit phone number.");
      return;
    }

    try {
      console.log(`Attempting to register with email: ${userData.email}`);
      console.log('Sending data:', JSON.stringify(userData, null, 2));
      const response = await fetch('/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(userData)
      });

      const data = await response.json();
      console.log('Server response:', data);

      if (response.ok) {
        try {
          // Sign in the user with email and password
          console.log(`Attempting to sign in with email: ${userData.email}`);
          const userCredential = await firebase.auth().signInWithEmailAndPassword(userData.email, userData.password);
          const user = userCredential.user;
          if (user) {
            const idToken = await user.getIdToken(true);
            console.log('Firebase ID Token:', idToken.substring(0, 20) + '...');
            sessionStorage.setItem('idToken', idToken);

            // Store ID token in a cookie to persist across navigation
            document.cookie = `idToken=${idToken}; path=/; max-age=3600; SameSite=Strict`;

            alert(`${role} registered successfully! Please continue with the next steps.`);

            if (data.redirect === '/further_patient_registration') {
              console.log('Fetching /further_patient_registration with ID token');
              const furtherResponse = await fetch('/further_patient_registration', {
                method: 'GET',
                headers: {
                  'Authorization': `Bearer ${idToken}`
                }
              });
              if (furtherResponse.ok) {
                console.log('Successfully accessed /further_patient_registration');
                const html = await furtherResponse.text();
                // Render the response directly to avoid browser navigation
                document.open();
                document.write(html);
                document.close();
              } else {
                const furtherData = await furtherResponse.json();
                console.error('Failed to access /further_patient_registration:', furtherData);
                alert(`Error accessing further registration: ${furtherData.error || 'Unknown error'}`);
                window.location.href = '/login';
              }
            } else {
              window.location.href = data.redirect || '/dashboard';
            }
          } else {
            throw new Error('Firebase user authentication failed');
          }
        } catch (firebaseError) {
          console.error('Firebase Authentication Error:', firebaseError);
          alert('Registration succeeded, but authentication failed. Please try logging in manually.');
          window.location.href = '/login';
        }
      } else {
        console.error('Registration failed:', data.error);
        alert(data.error || 'Registration failed');
      }
    } catch (error) {
      console.error("Registration error:", error);
      alert("Registration failed: " + (error.message || error));
    }
  });
});