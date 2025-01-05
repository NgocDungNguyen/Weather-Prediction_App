document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const results = document.getElementById('results');
    const loading = document.getElementById('loading');
    const errorModal = document.getElementById('error-modal');
    const errorMessage = document.getElementById('error-message');
    const closeError = document.getElementById('close-error');
    const fileUpload = document.getElementById('file-upload');
    const fileLabel = document.querySelector('label[for="file-upload"]');

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        loading.classList.remove('hidden');
        results.classList.add('hidden');

        const formData = new FormData(form);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response =&gt; response.json())
        .then(data =&gt; {
            loading.classList.add('hidden');
            if (data.error) {
                showError(data.error);
            } else {
                displayResults(data);
            }
        })
        .catch(error =&gt; {
            loading.classList.add('hidden');
            showError('An error occurred. Please try again.');
        });
    });

    function displayResults(data) {
        results.classList.remove('hidden');
        const table = document.getElementById('prediction-table');
        table.innerHTML = '
<table class="w-full border-collapse border border-gray-300">
 <thead>
  <tr class="bg-gray-100">
   <th class="border border-gray-300 px-4 py-2">
    Date
   </th>
   <th class="border border-gray-300 px-4 py-2">
    Predicted Max Temperature (°C)
   </th>
  </tr>
 </thead>
 <tbody>
 </tbody>
</table>
';
        const tbody = table.querySelector('tbody');

        data.predictions.forEach(prediction =&gt; {
            const row = document.createElement('tr');
            row.innerHTML = `
<td class="border border-gray-300 px-4 py-2">
 ${prediction.date}
</td>
<td class="border border-gray-300 px-4 py-2">
 ${prediction.predicted_tempmax.toFixed(2)}
</td>
`;
            tbody.appendChild(row);
        });

        document.getElementById('temp-over-time').src = '/static/outputs/temperature_over_time.png';
        document.getElementById('temp-distribution').src = '/static/outputs/temperature_distribution.png';
        document.getElementById('correlation-heatmap').src = '/static/outputs/correlation_heatmap.png';
        document.getElementById('download-csv').href = `/download/${data.csv_filename}`;

        results.scrollIntoView({ behavior: 'smooth' });
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorModal.classList.remove('hidden');
    }

    closeError.addEventListener('click', function() {
        errorModal.classList.add('hidden');
    });

    fileUpload.addEventListener('change', function(e) {
        if (e.target.files.length &gt; 0) {
            fileLabel.textContent = e.target.files[0].name;
        } else {
            fileLabel.textContent = 'Upload a file';
        }
    });

    const contactForm = document.getElementById('contact-form');
    contactForm.addEventListener('submit', function(e) {
        e.preventDefault();
        alert('Thank you for your message. We will get back to you soon!');
        contactForm.reset();
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor =&gt; {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
});