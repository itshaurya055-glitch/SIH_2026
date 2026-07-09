FROM node:22-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

COPY . .

# Expose Vite's default port
EXPOSE 5173

# Run the dev server and expose it on the network
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
