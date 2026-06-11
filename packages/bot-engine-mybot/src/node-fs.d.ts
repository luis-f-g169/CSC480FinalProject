declare module "node:fs" {
    export function writeFileSync(file: URL | string, data: string): void;
}
