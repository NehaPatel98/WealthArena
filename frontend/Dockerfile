FROM node:18-bullseye

WORKDIR /app

ENV NODE_ENV=development \
    CI=false

RUN npm i -g expo-cli@6.x

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 19000 19001 19002 5001

CMD ["npx", "expo", "start", "--tunnel", "--host", "0.0.0.0"]


