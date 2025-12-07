---
layout: default
title: Misc
permalink: /misc/
---

Hidden articles. They're just not interesting, but I wrote them up, and sometimes still use them to refresh my memory right before interviews. They're just not cool enough for me to advertise on the main page.

<h1>Miscellaneous Notes</h1>

<div id="misc-posts">
  {% for post in site.categories.misc %}
    <div class="post-row">
      <div class="post-title">
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      </div>
      {% if post.subtitle %}
        <div class="post-subtitle">{{ post.subtitle }}</div>
      {% endif %}
    </div>
  {% endfor %}
</div>