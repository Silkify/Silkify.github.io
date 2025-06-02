---
title: "Project"
layout: archive
permalink: /categories/Project/
author_profile: true
entries_layout: list
toc: true
toc_sticky: true
taxonomy: Project

---


{% assign posts = site.categories.Project %}

{% assign posts_lower = site.categories.Project %}

{% for post in posts %}
  <h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
  <p>{{ post.excerpt }}</p>
{% endfor %}
