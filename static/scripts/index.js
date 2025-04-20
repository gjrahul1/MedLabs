(function () {
    // Canvas Setup
    const canvasContainer = document.getElementById('canvas-container');
    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 400;
    canvasContainer.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    // Heartbeat Line Animation
    let time = 0;
    const speed = 0.05;
    const amplitude = 50;
    const frequency = 0.1;
    const points = [];
    const numPoints = 200;

    // Initialize points for the heartbeat line
    for (let i = 0; i < numPoints; i++) {
        points.push({
            x: (i / numPoints) * canvas.width,
            y: canvas.height / 2
        });
    }

    function drawHeartbeatLine() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        ctx.strokeStyle = '#ff5555';
        ctx.lineWidth = 2;

        // Simulate heartbeat pattern
        for (let i = 0; i < points.length; i++) {
            const point = points[i];
            const offset = i - (time % points.length);
            const heartbeat = Math.sin(offset * frequency) * amplitude;

            // Add heartbeat spikes (P, QRS, T waves simplified)
            if (offset % 50 < 5) { // QRS spike
                point.y = canvas.height / 2 - heartbeat * 3;
            } else if (offset % 50 < 10) { // P wave
                point.y = canvas.height / 2 - heartbeat * 0.5;
            } else if (offset % 50 < 15) { // T wave
                point.y = canvas.height / 2 - heartbeat * 0.3;
            } else {
                point.y = canvas.height / 2 + heartbeat * 0.1;
            }

            if (i === 0) {
                ctx.moveTo(point.x, point.y);
            } else {
                ctx.lineTo(point.x, point.y);
            }
        }

        ctx.stroke();
        time += speed;
        requestAnimationFrame(drawHeartbeatLine);
    }

    // Start Animation
    drawHeartbeatLine();

    // Responsive Resizing
    window.addEventListener('resize', () => {
        canvas.width = 400;
        canvas.height = 400;
    });

    // Testimonial Carousel
    let currentTestimonial = 0;
    const testimonials = document.querySelectorAll('.testimonial-item');
    const dots = document.querySelectorAll('.nav-dot');

    function showTestimonial(index) {
        testimonials[currentTestimonial].classList.remove('active');
        dots[currentTestimonial].classList.remove('active');
        currentTestimonial = index;
        testimonials[currentTestimonial].classList.add('active');
        dots[currentTestimonial].classList.add('active');
    }

    // Assign to window for onclick handlers in HTML
    window.showTestimonial = showTestimonial;

    // Auto-scroll Testimonials
    setInterval(() => {
        let next = currentTestimonial + 1;
        if (next >= testimonials.length) next = 0;
        showTestimonial(next);
    }, 5000);
})();