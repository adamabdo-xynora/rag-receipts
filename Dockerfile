# rag-receipts — multi-stage build.
#
# NO SECRETS ANYWHERE IN THIS FILE, AND NONE MAY BE ADDED. No ARG or ENV for
# API keys, no `.env` copied in — anything baked into a layer is readable by
# anyone who can pull the image. Keys reach a container only at run time, via
# `docker run -e` or `--env-file`. This repository's whole discipline is that
# a key enters through exactly one read — `process.env` at the bottom of
# src/eval-cli.ts — and the image must not become a second route.
#
# node:22 matches the major version CI pins (.github/workflows/ci.yml,
# node-version: 22). The -slim variant is used because the full offline suite
# (npm test, npm run typecheck, npm run demo) has been verified to pass on it.

# deps: install from the manifest and lockfile only, so this layer is
# invalidated by dependency changes, never by source changes.
FROM node:22-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# test: devDependencies plus the full source, corpus, eval, demo, and test
# trees. Everything it runs is offline by construction — no keys:
#
#   docker build --target test -t rag-receipts:test .
#   docker run --rm rag-receipts:test npm test           # the full suite
#   docker run --rm rag-receipts:test npm run typecheck
#   docker run --rm rag-receipts:test npm run demo
FROM deps AS test
COPY tsconfig.json ./
COPY src/ ./src/
COPY corpus/ ./corpus/
COPY eval/ ./eval/
COPY demo/ ./demo/
COPY test/ ./test/
CMD ["npm", "test"]

# build: compile src/ to plain ESM JavaScript so the runtime stage needs
# neither tsx nor typescript (both devDependencies). tsconfig's `include`
# spans src, test, and demo; only dist/src is carried forward.
FROM test AS build
RUN npx tsc -p tsconfig.json --outDir dist

# runtime (default): production dependencies and the compiled eval CLI, plus
# the corpus and question set it reads. This stage deliberately cannot run
# vitest — shipping test tooling in a runtime image is surface area for no
# benefit; the test stage above exists precisely so nothing tempts anyone to
# fatten this one.
FROM node:22-slim AS runtime
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev
# The compiled CLI sits at the same depth as in the repo, so its
# import.meta-relative paths (../corpus, ../eval, ../results) resolve
# unchanged.
COPY --from=build /app/dist/src/ ./src/
COPY corpus/ ./corpus/
COPY eval/ ./eval/
# Keys are supplied at run time only, e.g.:
#   docker run --rm -e ANTHROPIC_API_KEY -e VOYAGE_API_KEY rag-receipts
ENTRYPOINT ["node", "src/eval-cli.js"]
