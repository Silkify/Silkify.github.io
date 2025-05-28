---
layout: home
title: "Study shop"
---
# Welcome to Study shop
This blog focuses on Mathematical Statistics and Data Analysis projects.

## 수리통계학 프로젝트
[수리통계학 페이지 바로가기](/categories/수리통계학/)

### Recent Posts
{% for post in site.posts limit:3 %}
  <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
  <p>{{ post.date | date: "%Y.%m.%d" }} - {{ post.excerpt }}</p>
{% endfor %}