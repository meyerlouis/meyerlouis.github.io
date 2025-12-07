---
layout: default
title: Blog
---

Here are some <a href="/misc/" style="color:inherit; text-decoration:none;">articles I wrote</a>. I wrote them for myself, as I find these subjects interesting. They also tend to come up in interviews for ML / quant positions, so now I can access my notes online from my phone, while waiting for my interviews at the onsites. 

Feel free to send me an email if you believe I made a mistake or have omitted any important details.

These notes are made to be easily readible. It assumes that the reader is familiar with the basic subjects as I rarely explain the terminology or the motivation. I try to make these posts as practical as possible and as information-dense as possible, while introducing concepts intuitively. 

Most of the content is generally coming directly from (no more than) a handful of sources, which I do not hesitate to paraphrase from if I like the wording. Even though this is the case, I try to bring a new flavour, add explanations or proofs to the concepts I explore.

<hr class="post-divider">

<div id="posts">
  {% for post in site.posts %}
    {% unless post.categories contains "misc" %}
    <div class="post-row">
      <div class="post-title">
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      </div>
      {% if post.subtitle %}
      <div class="post-subtitle">{{ post.subtitle }}</div>
      {% endif %}
      <!-- <div class="meta">{{ post.date | date: "%B %d, %Y" }}</div> -->
    </div>
    {% endunless %}
  {% endfor %}
</div>