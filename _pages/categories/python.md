---
title: "python"
layout: archive
permalink: /categories/python/
author_profile: true
entries_layout: list
toc: true
toc_sticky: true
taxonomy: python

---


{% assign posts = site.categories.python %}

{% assign posts_lower = site.categories.python %}

{% for post in posts %}
  <h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
  <p>{{ post.excerpt }}</p>
{% endfor %}
