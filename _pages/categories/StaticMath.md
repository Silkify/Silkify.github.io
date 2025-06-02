---
title: "StaticMath"
layout: archive
permalink: /categories/StaticMath/
author_profile: true
entries_layout: list
toc: true
toc_sticky: true
taxonomy: StaticMath

---


{% assign posts = site.categories.StaticMath %}

{% assign posts_lower = site.categories.StaticMath %}

{% for post in posts %}
  <h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
  <p>{{ post.excerpt }}</p>
{% endfor %}


