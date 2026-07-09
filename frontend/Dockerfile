FROM nginx:alpine

# Copy all frontend static files into nginx's document root
COPY index.html /usr/share/nginx/html/
COPY terrain.html /usr/share/nginx/html/

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
