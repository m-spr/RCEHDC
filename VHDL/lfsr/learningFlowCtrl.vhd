library IEEE;
  use IEEE.STD_LOGIC_1164.all;
  use IEEE.NUMERIC_STD.all;

entity learningFlowCtrl is
  generic (lgCn : INTEGER := 4);
  port (
    clk, rst     : in  STD_LOGIC;
    learning     : in  STD_LOGIC;
    TVALID_SI    : in  STD_LOGIC;
    TLAST_SI     : in  STD_LOGIC;
    classIndexI  : in  STD_LOGIC_VECTOR(lgCn - 1 downto 0);
    ground_truth : in  INTEGER;
    update_done  : in  STD_LOGIC;
    learningRun  : out STD_LOGIC;
    TLAST_S      : out STD_LOGIC;
    TVALID_S     : out STD_LOGIC
  );
end entity;

architecture ctrl of learningFlowCtrl is
  signal learningRunI : STD_LOGIC := '0';
begin

  process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        learningRunI <= '0';
        TLAST_S <= '0';
        TVALID_S <= '0';
      elsif learningRunI = '1' then
        TLAST_S <= '0';
        TVALID_S <= '0';
        if update_done = '1' then
          learningRunI <= '0';
          TLAST_S <= '1';
          TVALID_S <= '1';
        end if;
      elsif learning = '1' and TVALID_SI = '1' and to_integer(unsigned(classIndexI)) /= ground_truth then
        learningRunI <= '1';
        TLAST_S <= '0';
        TVALID_S <= '0';
      else
        learningRunI <= '0';
        TLAST_S <= TLAST_SI;
        TVALID_S <= TVALID_SI;
      end if;
    end if;
  end process;

  learningRun <= learningRunI;

end architecture;
