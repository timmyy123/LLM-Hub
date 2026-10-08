extern int lean_bridge_probe(void);
int dependency_value = 99;
int dependency_function(void) { return 100; }
int main(void) {
  return lean_bridge_probe() == 7 && dependency_value == 99 ? 0 : 1;
}
