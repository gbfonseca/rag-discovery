# RAG Discovery

Simple RAG implementation, to learn.

How to run:

Install all dependencies

```bash
npm i
```

Run the server

```bash
npm start
```

Do a request

```bash
curl --request POST \
  --url http://localhost:8080/prompt \
  --header 'Content-Type: application/json' \
  --header 'User-Agent: insomnia/10.3.0' \
  --data '{
	"question": "How many goals does Man United have as HomeTeam 1 or more?"
}'
```

Use a dataset of [Kaggle](https://www.kaggle.com/datasets/flynn28/european-football-matches)
