LIBRARY ieee;
    USE ieee.std_logic_1164.ALL;
    USE ieee.numeric_std.ALL;

ENTITY binarizer IS
    PORT (
        value  : IN  integer;
        result : OUT std_logic
    );
END ENTITY;

ARCHITECTURE behavioral OF binarizer IS
BEGIN
    result <= '1' WHEN value >= 0 ELSE '0';
END ARCHITECTURE;
