---
layout: default
title: Home
---

<div id="random-image-container" style="text-align: center; margin-top: 60px;">
  <!-- Filled by JavaScript -->
</div>

<script>
  document.addEventListener("DOMContentLoaded", function() {
    const items = {{ site.data.home_images | jsonify }};
    const chosen = items[Math.floor(Math.random() * items.length)];

    const container = document.getElementById("random-image-container");

    container.innerHTML = `
      <img src="${chosen.image}" 
           alt="Random image" 
           style="max-width: ${chosen.custom_width}; display: block; margin: 0 auto;">
       <p style="margin-top: 10px; font-size: 14px;">
       <i>${chosen.title}</i>, ${chosen.artist} (${chosen.date})
      </p>
    `;
  });
</script>
