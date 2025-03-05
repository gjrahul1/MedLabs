(function () {
    // Scene Setup
    let scene, camera, renderer;

    function initScene() {
        if (!scene) {
            scene = new THREE.Scene();
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setClearColor(0xffffff, 1);
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.getElementById('canvas-container').appendChild(renderer.domElement);
        }
    }

    // Stethoscope
    let stethoGeometry, stethoMaterial, stethoscope;
    function initStethoscope() {
        if (!stethoGeometry) {
            stethoGeometry = new THREE.TorusGeometry(5, 0.5, 16, 100, Math.PI * 1.5);
            stethoMaterial = new THREE.MeshBasicMaterial({ color: 0xff5555, wireframe: true });
            stethoscope = new THREE.Mesh(stethoGeometry, stethoMaterial);
            stethoscope.position.set(-20, 10, -50);
            scene.add(stethoscope);
        }
    }

    // AI Robot
    let robotGeometry, robotMaterial, robot, armGeometry, armMaterial, leftArm, rightArm;
    function initRobot() {
        if (!robotGeometry) {
            robotGeometry = new THREE.BoxGeometry(3, 5, 3);
            robotMaterial = new THREE.MeshBasicMaterial({ color: 0x00cc00, wireframe: true });
            robot = new THREE.Mesh(robotGeometry, robotMaterial);
            robot.position.set(0, 0, -50);
            scene.add(robot);

            armGeometry = new THREE.CylinderGeometry(0.3, 0.3, 4, 32);
            armMaterial = new THREE.MeshBasicMaterial({ color: 0x00cc00, wireframe: true });
            leftArm = new THREE.Mesh(armGeometry, armMaterial);
            leftArm.position.set(-2, 0, -50);
            leftArm.rotation.z = Math.PI / 4;
            scene.add(leftArm);

            rightArm = new THREE.Mesh(armGeometry, armMaterial);
            rightArm.position.set(2, 0, -50);
            rightArm.rotation.z = -Math.PI / 4;
            scene.add(rightArm);
        }
    }

    // Google Cloud Logo
    let cloudGeometry, cloudMaterial, cloud1, cloud2, cloud3;
    function initCloud() {
        if (!cloudGeometry) {
            cloudGeometry = new THREE.SphereGeometry(2, 32, 32);
            cloudMaterial = new THREE.MeshBasicMaterial({ color: 0x0078d4, wireframe: true });
            cloud1 = new THREE.Mesh(cloudGeometry, cloudMaterial);
            cloud1.position.set(20, 10, -50);
            scene.add(cloud1);
            cloud2 = new THREE.Mesh(cloudGeometry, cloudMaterial);
            cloud2.position.set(25, 8, -50);
            scene.add(cloud2);
            cloud3 = new THREE.Mesh(cloudGeometry, cloudMaterial);
            cloud3.position.set(22, 5, -50);
            scene.add(cloud3);
        }
    }

    // Camera Flash Effect
    function triggerFlash() {
        const flash = document.createElement('div');
        flash.className = 'flash';
        document.body.appendChild(flash);
        flash.style.animation = 'flashAnimation 0.3s ease';
        setTimeout(() => flash.remove(), 300);
    }

    setInterval(triggerFlash, 4000);

    const styleSheet = document.createElement('style');
    styleSheet.textContent = `
        @keyframes flashAnimation {
            0% { opacity: 0; }
            50% { opacity: 1; }
            100% { opacity: 0; }
        }
    `;
    document.head.appendChild(styleSheet);

    // Camera Positioning
    camera.position.z = 50;

    // Animation Loop
    function animate() {
        requestAnimationFrame(animate);
        if (stethoscope) {
            stethoscope.rotation.z += 0.02;
            stethoscope.position.y = 10 + Math.sin(Date.now() * 0.001) * 2;
        }
        if (robot) {
            robot.rotation.y += 0.01;
            leftArm.rotation.x = Math.sin(Date.now() * 0.002) * 0.3;
            rightArm.rotation.x = -Math.sin(Date.now() * 0.002) * 0.3;
        }
        if (cloud1) {
            cloud1.position.y = 10 + Math.sin(Date.now() * 0.001) * 1;
            cloud2.position.y = 8 + Math.cos(Date.now() * 0.0015) * 1;
            cloud3.position.y = 5 + Math.sin(Date.now() * 0.002) * 1;
        }
        if (renderer) {
            renderer.render(scene, camera);
        }
    }

    // Responsive Resizing
    window.addEventListener('resize', () => {
        if (camera && renderer) {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }
    });

    // Initialize All Components
    initScene();
    initStethoscope();
    initRobot();
    initCloud();
    animate();

    // Loader Animation (Realistic Caduceus Symbol)
    function initLoader() {
        const loader = document.getElementById('loader');
        const canvas = document.getElementById('loader-canvas');
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ canvas, alpha: true });
        renderer.setSize(200, 200);
        camera.position.z = 15;

        // Staff
        const staffGeometry = new THREE.CylinderGeometry(0.2, 0.2, 12, 32);
        const staffMaterial = new THREE.MeshBasicMaterial({ color: 0x0078d4, wireframe: true });
        const staff = new THREE.Mesh(staffGeometry, staffMaterial);
        scene.add(staff);

        // Realistic Snakes (Dynamic Path with Slithering)
        const snakePoints1 = [];
        const snakePoints2 = [];
        const segments = 20;
        for (let i = 0; i <= segments; i++) {
            const t = i / segments;
            const y = -6 + 12 * t;
            const x1 = Math.sin(t * Math.PI * 4 + time) * (1 - t * 0.5);
            const x2 = Math.sin(t * Math.PI * 4 + Math.PI + time) * (1 - t * 0.5);
            snakePoints1.push(new THREE.Vector3(x1, y, 0));
            snakePoints2.push(new THREE.Vector3(x2, y, 0));
        }

        const snakeCurve1 = new THREE.CatmullRomCurve3(snakePoints1);
        const snakeCurve2 = new THREE.CatmullRomCurve3(snakePoints2);
        const snakeGeometry1 = new THREE.TubeGeometry(snakeCurve1, 64, 0.3, 8, false);
        const snakeGeometry2 = new THREE.TubeGeometry(snakeCurve2, 64, 0.3, 8, false);
        const snakeMaterial = new THREE.MeshBasicMaterial({ color: 0x00cc00, wireframe: false });
        const snake1 = new THREE.Mesh(snakeGeometry1, snakeMaterial);
        const snake2 = new THREE.Mesh(snakeGeometry2, snakeMaterial);
        scene.add(snake1);
        scene.add(snake2);

        // Wings
        const wingGeometry = new THREE.PlaneGeometry(2, 1, 1);
        const wingMaterial = new THREE.MeshBasicMaterial({ color: 0xff5555, wireframe: true });
        const wing1 = new THREE.Mesh(wingGeometry, wingMaterial);
        const wing2 = new THREE.Mesh(wingGeometry, wingMaterial);
        wing1.position.set(-1.5, 6.5, 0);
        wing2.position.set(1.5, 6.5, 0);
        wing1.rotation.z = Math.PI / 4;
        wing2.rotation.z = -Math.PI / 4;
        scene.add(wing1);
        scene.add(wing2);

        // Slithering Animation
        let time = 0;
        function animateLoader() {
            requestAnimationFrame(animateLoader);
            time += 0.05;

            // Update snake paths for slithering effect
            for (let i = 0; i <= segments; i++) {
                const t = i / segments;
                const y = -6 + 12 * t;
                snakePoints1[i].x = Math.sin(t * Math.PI * 4 + time) * (1 - t * 0.5);
                snakePoints2[i].x = Math.sin(t * Math.PI * 4 + Math.PI + time) * (1 - t * 0.5);
            }
            snakeCurve1.points = snakePoints1;
            snakeCurve2.points = snakePoints2;
            snakeGeometry1.dispose();
            snakeGeometry2.dispose();
            snake1.geometry = new THREE.TubeGeometry(snakeCurve1, 64, 0.3, 8, false);
            snake2.geometry = new THREE.TubeGeometry(snakeCurve2, 64, 0.3, 8, false);

            // Rotate entire symbol
            staff.rotation.y += 0.03;
            snake1.rotation.y += 0.03;
            snake2.rotation.y += 0.03;
            wing1.rotation.y += 0.03;
            wing2.rotation.y += 0.03;

            renderer.render(scene, camera);
        }
        animateLoader();

        return { show: () => loader.classList.remove('hidden'), hide: () => loader.classList.add('hidden') };
    }

    // Button Click Handler
    document.getElementById('get-started').addEventListener('click', () => {
        const loader = initLoader();
        loader.show();
        setTimeout(() => {
            loader.hide();
            window.location.href = '/login';
        }, 5000);
    });
})();