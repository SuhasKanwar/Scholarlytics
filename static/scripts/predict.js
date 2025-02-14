document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("prediction-form");
  const result = document.getElementById("result");
  const loading = document.getElementById("loading");
  const predictedScore = document.getElementById("predicted-score");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    loading.classList.remove("hidden");
    result.classList.add("hidden");

    const formData = new FormData(form);
    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });
      
      if (response.ok) {
        const data = await response.json();
        predictedScore.textContent = data.prediction.toFixed(2);
        result.classList.remove("hidden");
      } else {
        throw new Error("Prediction failed");
      }
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred while making the prediction. Please try again.");
    } finally {
      loading.classList.add("hidden");
    }
  });
});